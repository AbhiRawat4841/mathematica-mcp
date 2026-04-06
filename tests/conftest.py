from __future__ import annotations

import functools
import os
import shutil
import sys
from pathlib import Path

import pytest

# Default routing memory to off during tests to prevent touching real ~/.cache.
# Override with MATHEMATICA_ROUTING_MEMORY=observe to run the full suite in observe mode.
os.environ.setdefault("MATHEMATICA_ROUTING_MEMORY", "off")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TESTS_ROOT = REPO_ROOT / "tests"

for path in (str(REPO_ROOT), str(SRC_ROOT), str(TESTS_ROOT)):
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

    result = _execute_via_wolframscript("1+1", timeout=30)
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
    # Corpus markers
    config.addinivalue_line("markers", "corpus: corpus-driven test case")
    config.addinivalue_line("markers", "tier_smoke: smoke-tier corpus test")
    config.addinivalue_line("markers", "tier_core: core-tier corpus test")
    config.addinivalue_line("markers", "tier_extended: extended-tier corpus test")
    config.addinivalue_line("markers", "tier_probe: probe-tier corpus test (non-blocking)")
    config.addinivalue_line("markers", "profile_math: requires math profile")
    config.addinivalue_line("markers", "profile_notebook: requires notebook profile")
    config.addinivalue_line("markers", "profile_full: requires full profile")
    config.addinivalue_line("markers", "needs_wolfram_runtime: requires wolfram runtime")
    config.addinivalue_line("markers", "needs_live_addon: requires live addon connection")
    config.addinivalue_line("markers", "needs_frontend: requires Mathematica frontend")
    config.addinivalue_line("markers", "needs_network: requires network connectivity")
    config.addinivalue_line("markers", "needs_resource: requires Wolfram resource system")
    config.addinivalue_line("markers", "needs_subkernels: requires parallel subkernels")
