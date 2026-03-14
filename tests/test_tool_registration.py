from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
LIST_TOOLS_SCRIPT = """
import asyncio, json
from mathematica_mcp.server import mcp
tools = asyncio.run(mcp.list_tools())
print(json.dumps(sorted(tool.name for tool in tools)))
"""


def _tool_names(env_overrides: dict[str, str]) -> list[str]:
    env = os.environ.copy()
    env.update(env_overrides)
    env["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", LIST_TOOLS_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    return json.loads(result.stdout)


def test_optional_tool_groups_can_be_disabled():
    baseline = _tool_names({})
    disabled = _tool_names(
        {
            "MATHEMATICA_ENABLE_LOOKUP": "false",
            "MATHEMATICA_ENABLE_MATH_ALIASES": "false",
            "MATHEMATICA_ENABLE_FUNCTION_REPO": "false",
            "MATHEMATICA_ENABLE_DATA_REPO": "false",
            "MATHEMATICA_ENABLE_ASYNC": "false",
            "MATHEMATICA_ENABLE_CACHE": "false",
            "MATHEMATICA_ENABLE_TELEMETRY": "false",
        }
    )

    assert "execute_code" in disabled
    assert len(disabled) < len(baseline)

    gated_tools = {
        "resolve_function",
        "get_symbol_info",
        "suggest_similar_functions",
        "mathematica_integrate",
        "search_function_repository",
        "search_data_repository",
        "submit_computation",
        "cache_expression",
        "get_telemetry_stats",
        "reset_telemetry",
    }
    assert gated_tools & set(baseline)
    assert not (gated_tools & set(disabled))


def test_telemetry_tools_only_register_when_enabled():
    disabled = _tool_names({"MATHEMATICA_ENABLE_TELEMETRY": "false"})
    enabled = _tool_names({"MATHEMATICA_ENABLE_TELEMETRY": "true"})

    assert "get_telemetry_stats" not in disabled
    assert "reset_telemetry" not in disabled
    assert "get_telemetry_stats" in enabled
    assert "reset_telemetry" in enabled
