"""G1 schema-budget gate (CI-gating).

Enforces the per-profile MCP tool-schema size budget from the v1.0 LEAN plan
(docs/roadmaps/v1.0-lean-plan-v2.md, gate G1). The ``lean`` profile is the hard
target: <= 18 tools and <= 16 KB of tool-schema JSON. Budgets for the existing
profiles act as regression ceilings that catch accidental schema bloat.

Token proxy: we gate on raw ``len(json.dumps(schema))`` bytes and treat
tokens ~= bytes / 4 (a documented approximation). 16 KB ~= 4k tokens, matching
the plan's lean objective. The committed baseline lives at
benchmarks/results/profile_surface_baseline.json (regenerate with
benchmarks/profile_surface.py); CHANGELOG cites those numbers.

Runs without Mathematica: it only imports the server and lists tool schemas.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mathematica_mcp.config import VALID_PROFILES

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

# (max_tools, max_schema_bytes). lean is the hard plan target; the rest are
# regression ceilings a few % above the committed baseline (math 22495,
# notebook 37903, full 60986) to catch unintended growth without flaking on
# small doc edits.
BUDGETS: dict[str, tuple[int, int]] = {
    "lean": (18, 16384),
    "math": (28, 23500),
    "notebook": (48, 39500),
    "full": (82, 63000),
    "classic": (82, 63000),
}

_MEASURE_SCRIPT = """
import asyncio, json
from mathematica_mcp.server import mcp
tools = asyncio.run(mcp.list_tools())
schema = [t.model_dump(mode="json") for t in tools]
print(json.dumps({
    "tool_count": len(tools),
    "schema_bytes": len(json.dumps(schema, sort_keys=True)),
}))
"""


def _measure(profile: str) -> dict[str, int]:
    """Import the server under a given profile in a clean subprocess and
    return its tool count and serialized tool-schema size in bytes."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MATHEMATICA_")}
    env["MATHEMATICA_PROFILE"] = profile
    env["PYTHONPATH"] = str(SRC_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", _MEASURE_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("profile", [p for p in BUDGETS if p in VALID_PROFILES])
def test_profile_within_schema_budget(profile: str) -> None:
    max_tools, max_bytes = BUDGETS[profile]
    surface = _measure(profile)

    assert surface["tool_count"] <= max_tools, f"{profile}: {surface['tool_count']} tools exceeds budget of {max_tools}"
    assert surface["schema_bytes"] <= max_bytes, (
        f"{profile}: schema is {surface['schema_bytes']} bytes "
        f"(~{surface['schema_bytes'] // 4} tok), exceeds budget of {max_bytes} "
        f"(~{max_bytes // 4} tok)"
    )


def test_every_valid_profile_has_a_budget() -> None:
    """Guard: no profile may ship without a declared schema budget — this is
    what forces the lean gate to activate the moment the lean profile lands."""
    missing = [p for p in VALID_PROFILES if p not in BUDGETS]
    assert not missing, f"profiles without a schema budget: {missing}"
