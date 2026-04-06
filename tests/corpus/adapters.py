"""Execution adapters for corpus test dispatch.

Each adapter calls actual MCP server tools with timeout enforcement.
The alias_codegen adapter replicates code-generation logic from
optional_math_aliases.py for tools registered as closures.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def _call_with_timeout(coro: Any, timeout: int) -> Any:
    return await asyncio.wait_for(coro, timeout=timeout)


def call_server_tool(tool: str, params: dict, timeout: int = 30) -> Any:
    """Call an MCP server tool by name. Primary execution path."""
    from mathematica_mcp import server

    func = getattr(server, tool)
    return asyncio.run(_call_with_timeout(func(**params), timeout))


def call_offline_notebook(tool: str, params: dict, timeout: int = 30) -> Any:
    """Call offline notebook-file reader tools."""
    from mathematica_mcp import server

    func = getattr(server, tool)
    return asyncio.run(_call_with_timeout(func(**params), timeout))


def call_live_frontend(tool: str, params: dict, timeout: int = 30) -> Any:
    """Call live frontend tools (create_notebook, write_cell, etc.)."""
    from mathematica_mcp import server

    func = getattr(server, tool)
    return asyncio.run(_call_with_timeout(func(**params), timeout))


def call_alias_codegen(tool: str, params: dict, timeout: int = 30) -> Any:
    """Math alias tools are closures (optional_math_aliases.py:8), not importable.

    This adapter replicates the code-generation logic and routes through
    execute_code. Tests mathematical correctness of code generation, NOT
    the registered closure wrapper.
    """
    from mathematica_mcp import server

    code = build_alias_code(tool, params)
    return asyncio.run(
        _call_with_timeout(
            server.execute_code(code=code, format="text", output_target="cli"),
            timeout,
        )
    )


ADAPTERS: dict[str, Any] = {
    "server_tool": call_server_tool,
    "offline_notebook_file": call_offline_notebook,
    "live_frontend": call_live_frontend,
    "alias_codegen": call_alias_codegen,
}


def build_alias_code(tool: str, params: dict) -> str:
    """Replicate code-gen from optional_math_aliases.py lines 9-98."""
    if tool == "mathematica_integrate":
        expr = params["expression"]
        var = params["variable"]
        lb = params.get("lower_bound")
        ub = params.get("upper_bound")
        if lb is not None and ub is not None:
            return f"Integrate[{expr}, {{{var}, {lb}, {ub}}}]"
        return f"Integrate[{expr}, {var}]"

    if tool == "mathematica_solve":
        eq = params["equation"]
        var = params["variable"]
        domain = params.get("domain")
        if domain:
            return f"Solve[{eq}, {var}, {domain}]"
        return f"Solve[{eq}, {var}]"

    if tool == "mathematica_simplify":
        expr = params["expression"]
        assumptions = params.get("assumptions")
        full = params.get("full", False)
        func = "FullSimplify" if full else "Simplify"
        if assumptions:
            return f"{func}[{expr}, Assumptions -> {assumptions}]"
        return f"{func}[{expr}]"

    if tool == "mathematica_differentiate":
        expr = params["expression"]
        var = params["variable"]
        order = params.get("order", 1)
        if order == 1:
            return f"D[{expr}, {var}]"
        return f"D[{expr}, {{{var}, {order}}}]"

    if tool == "mathematica_expand":
        return f"Expand[{params['expression']}]"

    if tool == "mathematica_factor":
        return f"Factor[{params['expression']}]"

    if tool == "mathematica_limit":
        expr = params["expression"]
        var = params["variable"]
        point = params["point"]
        direction = params.get("direction")
        if direction:
            return f'Limit[{expr}, {var} -> {point}, Direction -> "{direction}"]'
        return f"Limit[{expr}, {var} -> {point}]"

    if tool == "mathematica_series":
        expr = params["expression"]
        var = params["variable"]
        point = params.get("point", "0")
        order = params.get("order", 5)
        return f"Series[{expr}, {{{var}, {point}, {order}}}]"

    raise ValueError(f"Unknown alias tool: {tool}")
