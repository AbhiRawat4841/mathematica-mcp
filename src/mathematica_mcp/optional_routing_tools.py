"""Optional MCP tools for routing memory introspection (full profile only)."""

from __future__ import annotations

import json


def register_routing_tools(mcp) -> None:
    from . import routing_memory

    @mcp.tool()
    async def get_routing_memory_stats(include_hints: bool = False) -> str:
        """Get routing memory statistics: mode, cohort counts, top error families.

        Set include_hints=True to also compute and include routing hints (advise mode only).
        """
        instance = routing_memory._CURRENT_INSTANCE
        if instance is None:
            return json.dumps({"mode": "off", "reason": "init_failed"}, indent=2)
        stats = instance.get_stats()
        if include_hints and instance.mode == "advise":
            from .config import FEATURES

            stats["routing_hints"] = instance.get_routing_hints(FEATURES.profile)
        return json.dumps(stats, indent=2)

    @mcp.tool()
    async def clear_routing_memory() -> str:
        """Clear all routing memory stats and delete persisted data."""
        instance = routing_memory._CURRENT_INSTANCE
        if instance is None:
            return json.dumps({"success": False, "reason": "init_failed"}, indent=2)
        instance.clear()
        return json.dumps({"success": True, "message": "Routing memory cleared."}, indent=2)
