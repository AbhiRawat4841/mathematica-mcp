from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP


def register_telemetry_tools(
    mcp: FastMCP,
    *,
    get_usage_stats,
    reset_stats,
) -> None:
    @mcp.tool()
    async def get_telemetry_stats() -> str:
        """Get usage statistics for tools."""
        return json.dumps({"success": True, "tool_stats": get_usage_stats()}, indent=2)

    @mcp.tool()
    async def reset_telemetry() -> str:
        """Reset all telemetry statistics to zero."""
        reset_stats()
        return json.dumps({"success": True, "message": "Telemetry statistics reset"}, indent=2)
