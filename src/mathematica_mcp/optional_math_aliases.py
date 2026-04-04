from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP


def register_math_alias_tools(mcp: FastMCP, execute_code) -> None:
    @mcp.tool()
    async def mathematica_integrate(
        expression: str,
        variable: str,
        lower_bound: str | int | float | None = None,
        upper_bound: str | int | float | None = None,
    ) -> str:
        """Compute integral using Integrate."""
        if lower_bound is not None and upper_bound is not None:
            code = f"Integrate[{expression}, {{{variable}, {lower_bound}, {upper_bound}}}]"
        else:
            code = f"Integrate[{expression}, {variable}]"

        return await execute_code(code=code, format="text", output_target="cli")

    @mcp.tool()
    async def mathematica_solve(
        equation: str,
        variable: str,
        domain: str | None = None,
    ) -> str:
        """Solve an equation using Solve."""
        code = f"Solve[{equation}, {variable}, {domain}]" if domain else f"Solve[{equation}, {variable}]"

        return await execute_code(code=code, format="text", output_target="cli")

    @mcp.tool()
    async def mathematica_simplify(
        expression: str,
        assumptions: str | None = None,
        full: bool = False,
    ) -> str:
        """Simplify a mathematical expression."""
        func = "FullSimplify" if full else "Simplify"
        code = f"{func}[{expression}, Assumptions -> {assumptions}]" if assumptions else f"{func}[{expression}]"
        return await execute_code(code=code, format="text", output_target="cli")

    @mcp.tool()
    async def mathematica_differentiate(
        expression: str,
        variable: str,
        order: int = 1,
    ) -> str:
        """Compute derivative using D."""
        code = f"D[{expression}, {variable}]" if order == 1 else f"D[{expression}, {{{variable}, {order}}}]"
        return await execute_code(code=code, format="text", output_target="cli")

    @mcp.tool()
    async def mathematica_expand(expression: str) -> str:
        """Expand a mathematical expression."""
        return await execute_code(
            code=f"Expand[{expression}]",
            format="text",
            output_target="cli",
        )

    @mcp.tool()
    async def mathematica_factor(expression: str) -> str:
        """Factor a mathematical expression."""
        return await execute_code(
            code=f"Factor[{expression}]",
            format="text",
            output_target="cli",
        )

    @mcp.tool()
    async def mathematica_limit(
        expression: str,
        variable: str,
        point: str | int | float,
        direction: Literal["Left", "Right"] | None = None,
    ) -> str:
        """Compute limit using Limit."""
        if direction:
            code = f'Limit[{expression}, {variable} -> {point}, Direction -> "{direction}"]'
        else:
            code = f"Limit[{expression}, {variable} -> {point}]"

        return await execute_code(code=code, format="text", output_target="cli")

    @mcp.tool()
    async def mathematica_series(
        expression: str,
        variable: str,
        point: str | int | float = "0",
        order: int = 5,
    ) -> str:
        """Compute Taylor/power series expansion."""
        code = f"Series[{expression}, {{{variable}, {point}, {order}}}]"
        return await execute_code(code=code, format="text", output_target="cli")
