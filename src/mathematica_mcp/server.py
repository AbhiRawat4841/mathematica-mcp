import json
import logging
import os
import re
import difflib
from typing import Literal, Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP, Image

from .connection import get_mathematica_connection, close_connection
from .session import execute_in_kernel, get_kernel_session, close_kernel_session

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mathematica_mcp")

mcp = FastMCP("mathematica-mcp")


def _lookup_symbols_in_kernel(query: str) -> Dict[str, Any]:
    """Search for Wolfram symbols matching query via wolframscript."""
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return {"success": False, "error": "wolframscript not found"}

    lookup_code = f'''
Module[{{query, candidates, systemMatches, globalMatches, allMatches, getInfo}},
  query = "{query}";
  systemMatches = Select[Names["System`*"], StringContainsQ[#, query, IgnoreCase -> True] &];
  globalMatches = Select[Names["Global`*"], StringContainsQ[#, query, IgnoreCase -> True] &];
  allMatches = Take[Join[systemMatches, globalMatches], UpTo[20]];
  getInfo[sym_String] := Module[{{usage, opts, attrs, syntaxInfo}},
    usage = Quiet[Check[ToString[ToExpression[sym <> "::usage"]], ""]];
    opts = Quiet[Check[ToString[Length[Options[ToExpression[sym]]]], "0"]];
    attrs = Quiet[Check[ToString[Attributes[ToExpression[sym]]], "{{}}"]];
    syntaxInfo = Quiet[Check[ToString[SyntaxInformation[ToExpression[sym]]], "{{}}"]];
    <|"symbol" -> sym, "usage" -> usage, "options_count" -> opts, "attributes" -> attrs, "syntax_info" -> syntaxInfo|>
  ];
  <|"success" -> True, "query" -> query, "matches" -> (getInfo /@ allMatches)|>
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", lookup_code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()

        if result.returncode != 0:
            return {"success": False, "error": result.stderr or "Lookup failed"}

        return {"success": True, "raw_output": output}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Lookup timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _parse_wolfram_association(raw: str) -> Dict[str, Any]:
    """Convert Wolfram Association syntax (<|...|>) to Python dict."""
    try:
        s = raw.strip()
        s = re.sub(r"<\|", "{", s)
        s = re.sub(r"\|>", "}", s)
        s = re.sub(r"\s*->\s*", ": ", s)
        s = s.replace("True", "true").replace("False", "false")
        s = re.sub(r":\s*([A-Za-z][A-Za-z0-9`$]*)\s*([,}])", r': "\1"\2', s)
        return json.loads(s)
    except Exception:
        return {"success": True, "raw": raw, "parse_error": True}


def _extract_short_description(usage: str) -> str:
    """Extract first sentence from Wolfram usage string."""
    if not usage or usage == "Null":
        return "No description available"

    usage = re.sub(r"^[A-Za-z]+\[.*?\]\s*", "", usage, count=1)
    match = re.match(r"^([^.!?]*[.!?])", usage)
    if match:
        return match.group(1).strip()
    return usage[:100].strip() + ("..." if len(usage) > 100 else "")


def _extract_example_signature(usage: str, symbol: str) -> str:
    """Extract usage pattern like Symbol[args] from usage string."""
    if not usage or usage == "Null":
        return f"{symbol}[...]"

    pattern = rf"{re.escape(symbol)}\[[^\]]*\]"
    match = re.search(pattern, usage)
    if match:
        return match.group(0)
    return f"{symbol}[...]"


def _rank_candidates(query: str, candidates: List[Dict]) -> List[Dict]:
    """Rank symbol candidates by relevance using exact/prefix/similarity scoring."""
    query_lower = query.lower()
    scored = []

    for c in candidates:
        symbol = c.get("symbol", "")
        symbol_name = symbol.split("`")[-1]
        symbol_lower = symbol_name.lower()

        score = 0.0
        if symbol_lower == query_lower:
            score += 100
        elif symbol_lower.startswith(query_lower):
            score += 50
        elif query_lower in symbol_lower:
            score += 25

        similarity = difflib.SequenceMatcher(None, query_lower, symbol_lower).ratio()
        score += similarity * 20
        score -= len(symbol_name) * 0.1

        if symbol.startswith("System`"):
            score += 5

        c["_score"] = score
        c["symbol_name"] = symbol_name
        scored.append(c)

    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored


def _try_addon_command(command: str, params: Optional[dict] = None) -> dict:
    conn = get_mathematica_connection()
    return conn.send_command(command, params)


@mcp.tool()
async def get_mathematica_status() -> str:
    """
    Get the connection status and Mathematica system information.

    Returns information about frontend version, kernel version,
    number of open notebooks, and connection mode (addon vs kernel-only).
    """
    try:
        result = _try_addon_command("get_status")
        result["connection_mode"] = "addon"
        return json.dumps(result, indent=2)
    except Exception as e:
        try:
            session = get_kernel_session()
            from wolframclient.language import wlexpr

            version = session.evaluate(wlexpr("$VersionNumber"))
            return json.dumps(
                {
                    "connection_mode": "kernel_only",
                    "kernel_version": float(version),
                    "note": "Addon not running - notebook control unavailable. Execute StartMCPServer[] in Mathematica.",
                    "error": str(e),
                },
                indent=2,
            )
        except Exception as e2:
            return json.dumps(
                {
                    "connection_mode": "disconnected",
                    "error": f"No connection available: {e2}",
                },
                indent=2,
            )


@mcp.tool()
async def get_notebooks() -> str:
    """
    List all open Mathematica notebooks.

    Returns ID, filename, title, and modified status for each notebook.
    Use the ID to reference notebooks in other commands.
    """
    result = _try_addon_command("get_notebooks")
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_notebook_info(notebook: Optional[str] = None) -> str:
    """
    Get detailed information about a notebook.

    Args:
        notebook: Notebook ID or filename substring. If None, uses the currently selected notebook.

    Returns filename, directory, cell count, cell styles, and modification status.
    """
    result = _try_addon_command("get_notebook_info", {"notebook": notebook})
    return json.dumps(result, indent=2)


@mcp.tool()
async def create_notebook(title: str = "Untitled") -> str:
    """
    Create a new empty Mathematica notebook.

    Args:
        title: Window title for the new notebook

    Returns the notebook ID for use in subsequent commands.
    """
    result = _try_addon_command("create_notebook", {"title": title})
    return json.dumps(result, indent=2)


@mcp.tool()
async def save_notebook(
    notebook: Optional[str] = None,
    path: Optional[str] = None,
    format: Literal["Notebook", "PDF", "HTML", "TeX"] = "Notebook",
) -> str:
    """
    Save a notebook to disk.

    Args:
        notebook: Notebook ID or filename. If None, uses selected notebook.
        path: File path for saving. If None, saves to current location.
        format: Export format (Notebook, PDF, HTML, TeX)
    """
    result = _try_addon_command(
        "save_notebook", {"notebook": notebook, "path": path, "format": format}
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def close_notebook(notebook: Optional[str] = None) -> str:
    """
    Close a notebook.

    Args:
        notebook: Notebook ID or filename. If None, closes selected notebook.
    """
    result = _try_addon_command("close_notebook", {"notebook": notebook})
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_cells(notebook: Optional[str] = None, style: Optional[str] = None) -> str:
    """
    Get list of cells in a notebook.

    Args:
        notebook: Notebook ID. If None, uses selected notebook.
        style: Filter by cell style (e.g., "Input", "Output", "Text", "Section")

    Returns cell IDs, styles, and content previews.
    """
    result = _try_addon_command("get_cells", {"notebook": notebook, "style": style})
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_cell_content(cell_id: str) -> str:
    """
    Get the full content of a specific cell.

    Args:
        cell_id: The cell object ID (from get_cells output)
    """
    result = _try_addon_command("get_cell_content", {"cell_id": cell_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def write_cell(
    content: str,
    style: str = "Input",
    notebook: Optional[str] = None,
    position: Literal["After", "Before", "End", "Beginning"] = "After",
) -> str:
    """
    Write a new cell to a notebook.

    Args:
        content: Wolfram Language code or text to write
        style: Cell style - Input, Output, Text, Section, Subsection, Title, Code, etc.
        notebook: Target notebook ID. If None, uses selected notebook.
        position: Where to insert - After (after cursor), Before, End (of notebook), Beginning

    Examples:
        write_cell("Plot[Sin[x], {x, 0, 2 Pi}]")
        write_cell("Analysis Results", style="Section", position="Beginning")
        write_cell("This is explanatory text.", style="Text")
    """
    result = _try_addon_command(
        "write_cell",
        {
            "notebook": notebook,
            "content": content,
            "style": style,
            "position": position,
        },
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_cell(cell_id: str) -> str:
    """
    Delete a cell from a notebook.

    Args:
        cell_id: The cell object ID to delete
    """
    result = _try_addon_command("delete_cell", {"cell_id": cell_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def evaluate_cell(cell_id: str) -> str:
    """
    Evaluate a specific cell in its notebook.

    Args:
        cell_id: The cell object ID to evaluate
    """
    result = _try_addon_command("evaluate_cell", {"cell_id": cell_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def execute_code(
    code: str,
    format: Literal["text", "latex", "mathematica"] = "text",
    output_target: Literal["cli", "notebook"] = "notebook",
) -> str:
    """
    Execute Wolfram Language code and return the result.

    Args:
        code: Wolfram Language code to execute
        format: Output format - text (default), latex (for equations), mathematica (InputForm)
        output_target: Where to display the output - "notebook" (insert into active notebook, default) or "cli" (return text)

    Examples:
        execute_code("Plot[Sin[x], {x,0,Pi}]")  # renders in notebook by default
        execute_code("Integrate[x^2, x]", output_target="cli")  # returns text
    """
    if output_target == "notebook":
        try:

            def run_notebook_sequence():
                _try_addon_command(
                    "write_cell",
                    {"content": code, "style": "Input", "position": "After"},
                )
                _try_addon_command(
                    "execute_code",
                    {"code": "SelectionMove[SelectedNotebook[], Previous, Cell]"},
                )
                _try_addon_command("execute_selection", {})

            try:
                # Attempt 1: Direct execution
                run_notebook_sequence()
                return json.dumps(
                    {"status": "executed_in_notebook", "code": code}, indent=2
                )
            except Exception:
                # Attempt 2: Auto-create notebook and retry
                logger.info(
                    "Notebook execution failed. Attempting to create new notebook."
                )
                _try_addon_command("create_notebook", {"title": "Untitled Analysis"})
                run_notebook_sequence()
                return json.dumps(
                    {
                        "status": "executed_in_notebook",
                        "code": code,
                        "note": "Created new notebook 'Untitled Analysis' as none was active.",
                    },
                    indent=2,
                )

        except Exception as e:
            logger.warning(
                f"Notebook execution failed completely: {e}. Falling back to CLI."
            )
            # Fallback to CLI execution
            try:
                result = _try_addon_command(
                    "execute_code", {"code": code, "format": format}
                )
                if isinstance(result, dict):
                    result["note"] = (
                        "Executed via CLI as notebook interface returned an error."
                    )
                    return json.dumps(result, indent=2)
                # If result isn't a dict, we wrap it
                return f"{result}\n(Note: Executed via CLI as notebook interface returned an error.)"
            except Exception:
                result = execute_in_kernel(code, format)
                result["execution_mode"] = "kernel_fallback"
                result["note"] = (
                    "Executed via CLI as notebook interface returned an error."
                )
                return json.dumps(result, indent=2)

    try:
        result = _try_addon_command("execute_code", {"code": code, "format": format})
        return json.dumps(result, indent=2)
    except Exception:
        result = execute_in_kernel(code, format)
        result["execution_mode"] = "kernel_fallback"
        return json.dumps(result, indent=2)


@mcp.tool()
async def evaluate_selection(notebook: Optional[str] = None) -> str:
    """
    Evaluate the currently selected cell(s) in a notebook.

    Args:
        notebook: Notebook ID. If None, uses selected notebook.
    """
    result = _try_addon_command("execute_selection", {"notebook": notebook})
    return json.dumps(result, indent=2)


@mcp.tool()
async def screenshot_notebook(
    notebook: Optional[str] = None, max_height: int = 2000
) -> Image:
    """
    Capture a screenshot of an entire notebook window.

    Args:
        notebook: Notebook ID. If None, uses selected notebook.
        max_height: Maximum height in pixels (prevents huge images)

    Returns the screenshot as an image that can be viewed directly.
    """
    result = _try_addon_command(
        "screenshot_notebook", {"notebook": notebook, "max_height": max_height}
    )

    image_path = result["path"]
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    os.remove(image_path)
    return Image(data=image_bytes, format="png")


@mcp.tool()
async def screenshot_cell(cell_id: str) -> Image:
    """
    Capture a screenshot of a specific cell's content and output.

    Useful for seeing plots, graphics, or formatted mathematical output.

    Args:
        cell_id: The cell object ID to screenshot
    """
    result = _try_addon_command("screenshot_cell", {"cell_id": cell_id})

    image_path = result["path"]
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    os.remove(image_path)
    return Image(data=image_bytes, format="png")


@mcp.tool()
async def rasterize_expression(expression: str, image_size: int = 400) -> Image:
    """
    Render a Wolfram Language expression as an image.

    Useful for visualizing plots, matrices, or formatted output without
    modifying any notebook.

    Args:
        expression: Wolfram Language expression to render
        image_size: Size of the resulting image in pixels

    Examples:
        rasterize_expression("Plot[Sin[x], {x, 0, 2 Pi}]")
        rasterize_expression("MatrixForm[{{1, 2}, {3, 4}}]")
        rasterize_expression("Graphics[Circle[]]", image_size=200)
    """
    result = _try_addon_command(
        "rasterize_expression", {"expression": expression, "image_size": image_size}
    )

    image_path = result["path"]
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    os.remove(image_path)
    return Image(data=image_bytes, format="png")


@mcp.tool()
async def select_cell(cell_id: str) -> str:
    """
    Select a cell in the notebook (moves cursor to it).

    Args:
        cell_id: The cell object ID to select
    """
    result = _try_addon_command("select_cell", {"cell_id": cell_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def scroll_to_cell(cell_id: str) -> str:
    """
    Scroll the notebook view to make a cell visible.

    Args:
        cell_id: The cell object ID to scroll to
    """
    result = _try_addon_command("scroll_to_cell", {"cell_id": cell_id})
    return json.dumps(result, indent=2)


@mcp.tool()
async def export_notebook(
    path: str,
    notebook: Optional[str] = None,
    format: Literal["PDF", "HTML", "TeX", "Markdown"] = "PDF",
) -> str:
    """
    Export a notebook to another format.

    Args:
        path: Destination file path
        notebook: Notebook ID. If None, uses selected notebook.
        format: Export format (PDF, HTML, TeX, Markdown)

    Example:
        export_notebook("~/Documents/analysis.pdf", format="PDF")
    """
    result = _try_addon_command(
        "export_notebook", {"notebook": notebook, "path": path, "format": format}
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def resolve_function(
    query: str,
    expression: Optional[str] = None,
    auto_execute: bool = True,
    max_candidates: int = 5,
    output_target: Literal["cli", "notebook"] = "cli",
) -> str:
    """
    Search for Wolfram Language functions and optionally auto-execute.

    Finds the closest matching function(s) to the query. If unambiguous and
    expression is provided, executes automatically. Otherwise returns candidates
    for user clarification.

    Args:
        query: Function name or partial name to search for (e.g., "Plot", "Integra")
        expression: Full Wolfram expression to execute if function resolves unambiguously
        auto_execute: If True (default), execute expression when match is unambiguous
        max_candidates: Maximum number of candidates to return (default 5)
        output_target: Where to execute if auto-executing ("cli" or "notebook")

    Returns:
        JSON with status ("resolved", "ambiguous", "not_found"), candidates, and execution result if applicable

    Examples:
        resolve_function("Plot") -> resolved, shows Plot usage
        resolve_function("Integra") -> ambiguous, shows Integrate, NIntegrate, etc.
        resolve_function("Plot", expression="Plot[Sin[x], {x, 0, Pi}]") -> resolves and executes
    """
    SCORE_THRESHOLD = 80
    SCORE_GAP_THRESHOLD = 15

    lookup_result = _lookup_symbols_in_kernel(query)

    if not lookup_result.get("success"):
        return json.dumps(
            {
                "status": "error",
                "error": lookup_result.get("error", "Lookup failed"),
                "query": query,
            },
            indent=2,
        )

    raw_output = lookup_result.get("raw_output", "")
    if not raw_output:
        return json.dumps(
            {
                "status": "not_found",
                "query": query,
                "message": f"No functions found matching '{query}'",
            },
            indent=2,
        )

    candidates_raw = []
    try:
        lines = raw_output.split("\n")
        for line in lines:
            if '"symbol"' in line and "->" in line:
                symbol_match = re.search(r'"symbol"\s*->\s*"([^"]+)"', line)
                usage_match = re.search(r'"usage"\s*->\s*"([^"]*)"', line)
                if symbol_match:
                    candidates_raw.append(
                        {
                            "symbol": symbol_match.group(1),
                            "usage": usage_match.group(1) if usage_match else "",
                        }
                    )
    except Exception:
        pass

    if not candidates_raw:
        try:
            import subprocess
            import shutil

            wolframscript = shutil.which("wolframscript")
            if wolframscript:
                simple_code = f'Names["*{query}*"]'
                result = subprocess.run(
                    [wolframscript, "-code", simple_code],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    names = re.findall(r'"([^"]+)"', result.stdout)
                    for name in names[:20]:
                        candidates_raw.append({"symbol": name, "usage": ""})
        except Exception:
            pass

    if not candidates_raw:
        return json.dumps(
            {
                "status": "not_found",
                "query": query,
                "message": f"No functions found matching '{query}'",
            },
            indent=2,
        )

    ranked = _rank_candidates(query, candidates_raw)
    top_candidates = ranked[:max_candidates]

    formatted_candidates = []
    for c in top_candidates:
        symbol_name = c.get("symbol_name", c.get("symbol", ""))
        usage = c.get("usage", "")
        formatted_candidates.append(
            {
                "symbol": symbol_name,
                "full_name": c.get("symbol", symbol_name),
                "description": _extract_short_description(usage),
                "example": _extract_example_signature(usage, symbol_name),
                "score": round(c.get("_score", 0), 2),
            }
        )

    is_resolved = False
    if len(formatted_candidates) >= 1:
        top_score = formatted_candidates[0]["score"]
        if top_score >= SCORE_THRESHOLD:
            if len(formatted_candidates) == 1:
                is_resolved = True
            else:
                second_score = formatted_candidates[1]["score"]
                if top_score - second_score >= SCORE_GAP_THRESHOLD:
                    is_resolved = True

    if is_resolved:
        resolved_symbol = formatted_candidates[0]
        response = {
            "status": "resolved",
            "query": query,
            "resolved_symbol": resolved_symbol["symbol"],
            "description": resolved_symbol["description"],
            "example": resolved_symbol["example"],
            "other_candidates": formatted_candidates[1:]
            if len(formatted_candidates) > 1
            else [],
        }

        if auto_execute and expression:
            exec_result = await execute_code(
                code=expression, format="text", output_target=output_target
            )
            response["execution"] = {
                "executed": True,
                "expression": expression,
                "result": json.loads(exec_result)
                if exec_result.startswith("{")
                else exec_result,
            }

        return json.dumps(response, indent=2)

    return json.dumps(
        {
            "status": "ambiguous",
            "query": query,
            "message": f"Multiple functions match '{query}'. Please clarify which one you need.",
            "candidates": formatted_candidates,
            "hint": "Provide more specific query or select from candidates above",
        },
        indent=2,
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
