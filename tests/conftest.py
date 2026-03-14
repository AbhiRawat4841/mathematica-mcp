from __future__ import annotations

import functools
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@functools.lru_cache(maxsize=1)
def wolfram_runtime_available() -> bool:
    """Return whether wolframscript can execute a trivial expression."""
    if shutil.which("wolframscript") is None:
        return False

    try:
        from mathematica_mcp.session import _execute_via_wolframscript
    except Exception:
        return False

    result = _execute_via_wolframscript("1+1", timeout=5)
    return bool(result.get("success")) and result.get("output_inputform") == "2"


@pytest.fixture
def require_wolfram_runtime() -> None:
    if not wolfram_runtime_available():
        pytest.skip("wolframscript runtime is not available in this environment")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "wolfram_runtime: requires a working wolframscript/Wolfram runtime",
    )
