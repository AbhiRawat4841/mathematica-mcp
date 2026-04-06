"""Optional MCP tools for routing memory introspection (full profile only)."""

from __future__ import annotations

import json


def register_routing_tools(mcp) -> None:
    from . import routing_memory

    @mcp.tool()
    async def get_routing_memory_stats() -> str:
        """Get routing memory statistics: mode, cohort counts, top error families."""
        instance = routing_memory._CURRENT_INSTANCE
        if instance is None:
            return json.dumps({"mode": "off", "reason": "init_failed"}, indent=2)
        return json.dumps(instance.get_stats(), indent=2)

    @mcp.tool()
    async def clear_routing_memory() -> str:
        """Clear all routing memory stats and delete persisted data."""
        instance = routing_memory._CURRENT_INSTANCE
        if instance is None:
            return json.dumps({"success": False, "reason": "init_failed"}, indent=2)
        instance.clear()
        return json.dumps({"success": True, "message": "Routing memory cleared."}, indent=2)
