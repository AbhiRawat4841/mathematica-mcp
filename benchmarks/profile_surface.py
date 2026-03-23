from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mathematica_mcp.config import VALID_PROFILES, FeatureFlags  # noqa: E402


@dataclass
class ProfileSurface:
    profile: str
    tool_count: int
    schema_bytes: int
    tool_names: list[str]


async def _collect_profile(profile: str) -> ProfileSurface:
    os.environ["MATHEMATICA_PROFILE"] = profile

    import importlib

    import mathematica_mcp.config as config
    import mathematica_mcp.server as server

    importlib.reload(config)
    server = importlib.reload(server)

    tools = await server.mcp.list_tools()
    tool_names = sorted(tool.name for tool in tools)
    schema_bytes = len(json.dumps([tool.model_dump(mode="json") for tool in tools], sort_keys=True))
    return ProfileSurface(
        profile=profile,
        tool_count=len(tool_names),
        schema_bytes=schema_bytes,
        tool_names=tool_names,
    )


async def main() -> None:
    report = {
        "profiles": [asdict(await _collect_profile(profile)) for profile in VALID_PROFILES],
        "defaults": FeatureFlags.from_env().to_dict(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
