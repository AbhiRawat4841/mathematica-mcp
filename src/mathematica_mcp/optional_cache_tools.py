from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP


def register_cache_tools(
    mcp: FastMCP,
    *,
    cache_expression_fn,
    get_cached_expression_fn,
    list_cached_expressions_fn,
    clear_cache_fn,
    execute_code,
) -> None:
    @mcp.tool()
    async def cache_expression(name: str, expression: str) -> str:
        """Evaluate and cache a Wolfram expression for later reuse."""
        result = await execute_code(code=expression, format="text", output_target="cli")
        try:
            result_data = json.loads(result)
            output = result_data.get("output", result)
        except (json.JSONDecodeError, TypeError):
            output = result

        success = cache_expression_fn(name, expression, str(output))
        return json.dumps(
            {
                "success": success,
                "name": name,
                "expression": expression,
                "result": str(output)[:500],
                "cached": success,
            },
            indent=2,
        )

    @mcp.tool()
    async def get_cached(name: str) -> str:
        """Retrieve a previously cached expression result."""
        cached = get_cached_expression_fn(name)
        if cached is None:
            return json.dumps(
                {"success": False, "error": f"No cached expression named '{name}'"},
                indent=2,
            )

        return json.dumps(
            {
                "success": True,
                "name": name,
                "expression": cached.expression,
                "result": cached.result,
                "access_count": cached.access_count,
            },
            indent=2,
        )

    @mcp.tool()
    async def list_cache() -> str:
        """List all cached expressions with their metadata."""
        cached = list_cached_expressions_fn()
        return json.dumps(
            {
                "success": True,
                "count": len(cached),
                "expressions": cached,
            },
            indent=2,
        )

    @mcp.tool()
    async def clear_expression_cache() -> str:
        """Clear all cached expressions."""
        clear_cache_fn()
        return json.dumps({"success": True, "message": "Expression cache cleared"}, indent=2)
