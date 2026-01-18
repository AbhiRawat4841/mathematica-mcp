import json
import logging
import os
import re
import difflib
from typing import Literal, Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP, Image

from .connection import get_mathematica_connection, close_connection
from .session import execute_in_kernel, get_kernel_session, close_kernel_session
from .config import FEATURES
from .telemetry import telemetry_tool, get_usage_stats, reset_stats
from .cache import (
    cache_expression as _cache_expr,
    get_cached_expression as _get_cached,
    list_cached_expressions,
    clear_cache,
    remove_cached_expression,
)

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
            if session is None:
                raise RuntimeError("No kernel session available")
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


@mcp.tool()
async def verify_derivation(
    steps: List[str],
    format: Literal["text", "latex", "mathematica"] = "text",
    timeout: int = 120,
) -> str:
    """
    Verify a sequence of mathematical expressions to check if each step
    logically follows from the previous one.

    Uses Mathematica's Simplify to check if consecutive expressions are equal.
    This is useful for verifying mathematical derivations or proofs.

    Args:
        steps: List of mathematical expressions (as strings) representing
               steps in a derivation. Requires at least two steps.
        format: Output format for the verification report:
            - "text" (default): Plain text report
            - "latex": LaTeX formatted expressions
            - "mathematica": Mathematica InputForm
        timeout: Maximum execution time in seconds (default: 120)

    Returns:
        A verification report showing which steps are valid and which fail.

    Examples:
        verify_derivation(["x^2 - y^2", "(x-y)*(x+y)"])
        -> "Step 1 -> 2: VALID (x^2 - y^2 == (x-y)*(x+y))"

        verify_derivation(["Sin[x]^2 + Cos[x]^2", "1"])
        -> "Step 1 -> 2: VALID (Pythagorean identity)"
    """
    import subprocess
    import shutil

    if len(steps) < 2:
        return json.dumps(
            {
                "success": False,
                "error": "At least two steps are required for verification",
                "steps_provided": len(steps),
            },
            indent=2,
        )

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    steps_list = ", ".join([f'"{step}"' for step in steps])

    if format == "latex":
        format_fn = "TeXForm"
    elif format == "mathematica":
        format_fn = "InputForm"
    else:
        format_fn = "ToString"

    verification_code = f"""
Module[{{steps, results, i, prev, current, isEqual, simplified, formatExpr}},
  steps = {{{steps_list}}};
  formatExpr = {format_fn};
  results = <|"success" -> True, "steps" -> {{}}, "summary" -> ""|>;
  
  Do[
    prev = ToExpression[steps[[i]]];
    current = ToExpression[steps[[i + 1]]];
    
    (* Check equivalence using multiple methods *)
    isEqual = Quiet[Check[
      TrueQ[Simplify[prev == current]] || 
      TrueQ[FullSimplify[prev == current]] ||
      TrueQ[Simplify[prev - current] == 0],
      False
    ]];
    
    simplified = Quiet[Check[Simplify[current], current]];
    
    AppendTo[results["steps"], <|
      "from" -> i,
      "to" -> i + 1,
      "expr_from" -> steps[[i]],
      "expr_to" -> steps[[i + 1]],
      "valid" -> isEqual,
      "simplified" -> ToString[formatExpr[simplified]]
    |>];
  , {{i, 1, Length[steps] - 1}}];
  
  (* Generate summary *)
  results["all_valid"] = AllTrue[results["steps"], #["valid"] &];
  results["valid_count"] = Count[results["steps"], _?(#["valid"] &)];
  results["total_steps"] = Length[steps] - 1;
  
  results
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", verification_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return json.dumps(
                {
                    "success": False,
                    "error": result.stderr or "Verification failed",
                    "stdout": result.stdout,
                },
                indent=2,
            )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)
        report_lines = ["## Derivation Verification Report\n"]

        if isinstance(parsed, dict) and "steps" in parsed:
            steps_data = parsed.get("steps", [])
            all_valid = parsed.get("all_valid", False)

            for step_info in steps_data:
                if isinstance(step_info, dict):
                    from_idx = step_info.get("from", "?")
                    to_idx = step_info.get("to", "?")
                    valid = step_info.get("valid", False)
                    expr_from = step_info.get("expr_from", "")
                    expr_to = step_info.get("expr_to", "")

                    status = "✓ VALID" if valid else "✗ INVALID"
                    report_lines.append(f"Step {from_idx} → {to_idx}: {status}")
                    report_lines.append(f"  From: {expr_from}")
                    report_lines.append(f"  To:   {expr_to}")
                    report_lines.append("")

            summary = (
                "All steps are valid!"
                if all_valid
                else "Some steps failed verification."
            )
            report_lines.append(f"**Summary**: {summary}")
            report_lines.append(
                f"Valid: {parsed.get('valid_count', 0)}/{parsed.get('total_steps', 0)} steps"
            )
        else:
            report_lines.append("Could not parse verification results.")
            report_lines.append(f"Raw output: {raw_output}")

        return json.dumps(
            {
                "success": True,
                "report": "\n".join(report_lines),
                "raw_data": parsed,
                "format": format,
            },
            indent=2,
        )

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "success": False,
                "error": f"Verification timed out after {timeout} seconds",
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Verification failed: {str(e)}"}, indent=2
        )


@mcp.tool()
async def get_symbol_info(symbol: str) -> str:
    """
    Get comprehensive information about a Wolfram Language symbol.

    Returns detailed introspection data including:
    - Full usage documentation
    - Options and their default values
    - Attributes (Listable, Protected, HoldAll, etc.)
    - Related functions
    - Example usage patterns
    - Syntax information

    Args:
        symbol: The Wolfram Language symbol name (e.g., "Plot", "Integrate", "Map")

    Returns:
        JSON with comprehensive symbol information

    Examples:
        get_symbol_info("Plot")
        get_symbol_info("Integrate")
        get_symbol_info("Map")
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    info_code = f"""
Module[{{sym, info, usage, opts, attrs, syntaxInfo, relatedSyms, examples}},
  sym = ToExpression["{symbol}"];
  
  (* Get usage string *)
  usage = Quiet[Check[
    ToString[sym::usage],
    "No usage information available"
  ]];
  
  (* Get options with defaults *)
  opts = Quiet[Check[
    Map[
      {{ToString[#[[1]]], ToString[#[[2]]]}} &,
      Options[sym]
    ],
    {{}}
  ]];
  
  (* Get attributes *)
  attrs = Quiet[Check[
    ToString /@ Attributes[sym],
    {{}}
  ]];
  
  (* Get syntax information *)
  syntaxInfo = Quiet[Check[
    SyntaxInformation[sym],
    {{}}
  ]];
  
  (* Get related symbols via WolframLanguageData if available *)
  relatedSyms = Quiet[Check[
    Take[
      ToString /@ WolframLanguageData["{symbol}", "RelatedSymbols"],
      UpTo[10]
    ],
    {{}}
  ]];
  
  (* Build result *)
  <|
    "success" -> True,
    "symbol" -> "{symbol}",
    "usage" -> usage,
    "options" -> opts,
    "options_count" -> Length[opts],
    "attributes" -> attrs,
    "syntax_info" -> ToString[syntaxInfo],
    "related_symbols" -> relatedSyms,
    "is_function" -> MemberQ[Attributes[sym], Protected],
    "context" -> Quiet[Check[Context[sym], "Unknown"]]
  |>
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", info_code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return json.dumps(
                {
                    "success": False,
                    "error": result.stderr or "Symbol lookup failed",
                    "symbol": symbol,
                },
                indent=2,
            )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        if isinstance(parsed, dict) and parsed.get("success"):
            formatted = {
                "success": True,
                "symbol": symbol,
                "usage": parsed.get("usage", ""),
                "attributes": parsed.get("attributes", []),
                "options_count": parsed.get("options_count", 0),
                "options": parsed.get("options", [])[:10],
                "related_symbols": parsed.get("related_symbols", []),
                "context": parsed.get("context", "Unknown"),
            }
            return json.dumps(formatted, indent=2)
        else:
            return json.dumps(
                {
                    "success": True,
                    "symbol": symbol,
                    "raw_output": raw_output,
                    "note": "Partial parsing - see raw output",
                },
                indent=2,
            )

    except subprocess.TimeoutExpired:
        return json.dumps(
            {"success": False, "error": "Symbol lookup timed out", "symbol": symbol},
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "symbol": symbol}, indent=2
        )


@mcp.tool()
async def get_kernel_state() -> str:
    """
    Get current Wolfram kernel session state information.

    Returns:
        - Defined user symbols in Global` context
        - Memory usage
        - Loaded packages
        - Session uptime (if available)
        - Kernel version

    Useful for understanding the current state of the computation environment.
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    state_code = """
<|
  "success" -> True,
  "kernel_version" -> $VersionNumber,
  "version_string" -> $Version,
  "system_id" -> $SystemID,
  "machine_name" -> $MachineName,
  "memory_in_use" -> MemoryInUse[],
  "memory_in_use_mb" -> Round[MemoryInUse[] / 1024.0 / 1024.0, 0.1],
  "max_memory_used" -> MaxMemoryUsed[],
  "loaded_packages" -> Quiet[Check[
    Select[Contexts[], StringMatchQ[#, __ ~~ "`"] && !StringStartsQ[#, "System`"] && !StringStartsQ[#, "Global`"] &],
    {}
  ]],
  "global_symbols" -> Quiet[Check[
    Take[Names["Global`*"], UpTo[50]],
    {}
  ]],
  "global_symbol_count" -> Length[Names["Global`*"]],
  "session_time" -> SessionTime[],
  "process_id" -> $ProcessID
|>
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", state_code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return json.dumps(
                {"success": False, "error": result.stderr or "State query failed"},
                indent=2,
            )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def load_package(package_name: str) -> str:
    """
    Load a Mathematica package into the current session.

    Args:
        package_name: The package name (e.g., "QuantumFramework`", "NeuralNetworks`")

    Returns:
        Success/failure status and list of new symbols made available.

    Examples:
        load_package("Developer`")
        load_package("GeneralUtilities`")
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    # Ensure package name ends with `
    if not package_name.endswith("`"):
        package_name = package_name + "`"

    load_code = f"""
Module[{{beforeContexts, afterContexts, newSymbols, result}},
  beforeContexts = Contexts[];
  
  result = Quiet[Check[
    Needs["{package_name}"];
    "loaded",
    "failed"
  ]];
  
  afterContexts = Contexts[];
  newSymbols = Complement[afterContexts, beforeContexts];
  
  <|
    "success" -> (result === "loaded"),
    "package" -> "{package_name}",
    "new_contexts" -> newSymbols,
    "message" -> If[result === "loaded", 
      "Package loaded successfully", 
      "Failed to load package"
    ]
  |>
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", load_code],
            capture_output=True,
            text=True,
            timeout=60,
        )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "package": package_name}, indent=2
        )


@mcp.tool()
async def list_loaded_packages() -> str:
    """
    List all currently loaded packages and contexts.

    Returns:
        List of loaded package contexts, excluding System` and Global`.
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    list_code = """
Module[{pkgs},
  pkgs = Select[
    Contexts[],
    StringMatchQ[#, __ ~~ "`"] && 
    !StringStartsQ[#, "System`"] && 
    !StringStartsQ[#, "Global`"] &&
    !StringStartsQ[#, "Internal`"] &
  ];
  <|
    "success" -> True,
    "packages" -> Sort[pkgs],
    "count" -> Length[pkgs]
  |>
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", list_code],
            capture_output=True,
            text=True,
            timeout=15,
        )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# ============================================================================
# Named Math Operation Aliases (more discoverable for LLMs)
# ============================================================================


@mcp.tool()
async def mathematica_integrate(
    expression: str,
    variable: str,
    lower_bound: Optional[str] = None,
    upper_bound: Optional[str] = None,
) -> str:
    """
    Compute integral using Mathematica's Integrate function.

    Args:
        expression: The expression to integrate (e.g., "x^2", "Sin[x]*Cos[x]")
        variable: Variable of integration (e.g., "x")
        lower_bound: Optional lower bound for definite integral
        upper_bound: Optional upper bound for definite integral

    Returns:
        The computed integral in text format

    Examples:
        mathematica_integrate("x^2", "x") -> "x^3/3"
        mathematica_integrate("x^2", "x", "0", "1") -> "1/3"
    """
    if lower_bound is not None and upper_bound is not None:
        code = f"Integrate[{expression}, {{{variable}, {lower_bound}, {upper_bound}}}]"
    else:
        code = f"Integrate[{expression}, {variable}]"

    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_solve(
    equation: str,
    variable: str,
    domain: Optional[str] = None,
) -> str:
    """
    Solve an equation using Mathematica's Solve function.

    Args:
        equation: The equation to solve (use == for equality, e.g., "x^2 - 4 == 0")
        variable: Variable to solve for (e.g., "x")
        domain: Optional domain constraint (e.g., "Reals", "Integers", "Complexes")

    Returns:
        Solutions in text format

    Examples:
        mathematica_solve("x^2 - 4 == 0", "x") -> "{{x -> -2}, {x -> 2}}"
        mathematica_solve("x^2 + 1 == 0", "x", "Reals") -> "{}"
    """
    if domain:
        code = f"Solve[{equation}, {variable}, {domain}]"
    else:
        code = f"Solve[{equation}, {variable}]"

    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_simplify(
    expression: str,
    assumptions: Optional[str] = None,
    full: bool = False,
) -> str:
    """
    Simplify a mathematical expression.

    Args:
        expression: The expression to simplify
        assumptions: Optional assumptions (e.g., "x > 0", "Element[n, Integers]")
        full: If True, use FullSimplify (more thorough but slower)

    Returns:
        Simplified expression

    Examples:
        mathematica_simplify("(x^2 - 1)/(x - 1)") -> "1 + x"
        mathematica_simplify("Sqrt[x^2]", assumptions="x > 0") -> "x"
    """
    func = "FullSimplify" if full else "Simplify"

    if assumptions:
        code = f"{func}[{expression}, Assumptions -> {assumptions}]"
    else:
        code = f"{func}[{expression}]"

    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_differentiate(
    expression: str,
    variable: str,
    order: int = 1,
) -> str:
    """
    Compute derivative using Mathematica's D function.

    Args:
        expression: Expression to differentiate
        variable: Variable to differentiate with respect to
        order: Order of derivative (default 1)

    Returns:
        The derivative

    Examples:
        mathematica_differentiate("x^3", "x") -> "3 x^2"
        mathematica_differentiate("Sin[x]", "x", 2) -> "-Sin[x]"
    """
    if order == 1:
        code = f"D[{expression}, {variable}]"
    else:
        code = f"D[{expression}, {{{variable}, {order}}}]"

    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_expand(expression: str) -> str:
    """
    Expand a mathematical expression.

    Args:
        expression: Expression to expand (e.g., "(x + 1)^3")

    Returns:
        Expanded expression

    Examples:
        mathematica_expand("(x + 1)^3") -> "1 + 3 x + 3 x^2 + x^3"
    """
    code = f"Expand[{expression}]"
    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_factor(expression: str) -> str:
    """
    Factor a mathematical expression.

    Args:
        expression: Expression to factor (e.g., "x^2 - 4")

    Returns:
        Factored expression

    Examples:
        mathematica_factor("x^2 - 4") -> "(-2 + x) (2 + x)"
    """
    code = f"Factor[{expression}]"
    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_limit(
    expression: str,
    variable: str,
    point: str,
    direction: Optional[Literal["Left", "Right"]] = None,
) -> str:
    """
    Compute limit using Mathematica's Limit function.

    Args:
        expression: Expression to take limit of
        variable: Variable approaching the limit
        point: Point to approach (can be "Infinity", "-Infinity", or a number)
        direction: Optional direction ("Left" or "Right" for one-sided limits)

    Returns:
        The limit value

    Examples:
        mathematica_limit("Sin[x]/x", "x", "0") -> "1"
        mathematica_limit("1/x", "x", "0", direction="Right") -> "Infinity"
    """
    if direction:
        code = f'Limit[{expression}, {variable} -> {point}, Direction -> "{direction}"]'
    else:
        code = f"Limit[{expression}, {variable} -> {point}]"

    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_series(
    expression: str,
    variable: str,
    point: str = "0",
    order: int = 5,
) -> str:
    """
    Compute Taylor/power series expansion.

    Args:
        expression: Expression to expand
        variable: Variable for expansion
        point: Point to expand around (default "0")
        order: Number of terms (default 5)

    Returns:
        Series expansion

    Examples:
        mathematica_series("Exp[x]", "x", "0", 5) -> "1 + x + x^2/2 + x^3/6 + x^4/24 + O[x]^5"
        mathematica_series("Sin[x]", "x") -> "x - x^3/6 + x^5/120 + O[x]^6"
    """
    code = f"Series[{expression}, {{{variable}, {point}, {order}}}]"
    result = await execute_code(code=code, format="text", output_target="cli")
    return result


# ============================================================================
# Wolfram Repository Integration
# ============================================================================


@mcp.tool()
async def search_function_repository(
    query: str,
    max_results: int = 10,
) -> str:
    """
    Search the Wolfram Function Repository for community functions.

    The Function Repository contains 2900+ community-contributed functions
    that extend Wolfram Language capabilities.

    Args:
        query: Search query (e.g., "graph", "image processing", "neural network")
        max_results: Maximum number of results to return (default 10)

    Returns:
        List of matching functions with names and descriptions

    Examples:
        search_function_repository("graph algorithms")
        search_function_repository("image segmentation")
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    search_code = f"""
Module[{{results, query}},
  query = "{query}";
  results = Quiet[Check[
    Take[
      ResourceSearch[{{"ResourceType" -> "Function", "Name" -> query}}, "SnippetData"],
      UpTo[{max_results}]
    ],
    {{}}
  ]];
  
  If[results === {{}},
    (* Fallback: try keyword search *)
    results = Quiet[Check[
      Take[
        ResourceSearch[{{"ResourceType" -> "Function", "Keyword" -> query}}, "SnippetData"],
        UpTo[{max_results}]
      ],
      {{}}
    ]]
  ];
  
  <|
    "success" -> True,
    "query" -> query,
    "count" -> Length[results],
    "results" -> Map[
      <|
        "name" -> #["Name"],
        "short_description" -> Quiet[Check[#["ShortDescription"], ""]],
        "repository_location" -> "Wolfram Function Repository"
      |> &,
      results
    ]
  |>
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", search_code],
            capture_output=True,
            text=True,
            timeout=60,
        )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except subprocess.TimeoutExpired:
        return json.dumps(
            {"success": False, "error": "Search timed out", "query": query}, indent=2
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "query": query}, indent=2)


@mcp.tool()
async def get_function_repository_info(function_name: str) -> str:
    """
    Get detailed information about a Wolfram Function Repository function.

    Args:
        function_name: The name of the function (e.g., "RandomGraph", "ImageSegmentation")

    Returns:
        Detailed function information including usage, examples, and documentation URL
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    info_code = f"""
Module[{{ro, info}},
  ro = Quiet[Check[
    ResourceObject["{function_name}"],
    $Failed
  ]];
  
  If[ro === $Failed,
    <|"success" -> False, "error" -> "Function not found in repository"|>,
    
    info = <|
      "success" -> True,
      "name" -> "{function_name}",
      "description" -> Quiet[Check[ro["Description"], ""]],
      "documentation_link" -> Quiet[Check[ro["DocumentationLink"], ""]],
      "version" -> Quiet[Check[ToString[ro["Version"]], ""]],
      "author" -> Quiet[Check[ro["ContributorInformation"], ""]],
      "keywords" -> Quiet[Check[ro["Keywords"], {{}}]],
      "usage_example" -> Quiet[Check[
        ToString[First[ro["BasicExamples"], ""]],
        ""
      ]]
    |>;
    info
  ]
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", info_code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "function": function_name}, indent=2
        )


@mcp.tool()
async def load_resource_function(function_name: str) -> str:
    """
    Load a function from the Wolfram Function Repository into the current session.

    Args:
        function_name: The name of the function to load

    Returns:
        Success status and basic usage information

    Example:
        load_resource_function("RandomGraph")
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    load_code = f"""
Module[{{fn, result}},
  fn = Quiet[Check[
    ResourceFunction["{function_name}"],
    $Failed
  ]];
  
  If[fn === $Failed,
    <|"success" -> False, "error" -> "Failed to load function from repository"|>,
    <|
      "success" -> True,
      "function" -> "{function_name}",
      "loaded" -> True,
      "usage" -> "Use ResourceFunction[\\"{function_name}\\"][args] to call the function",
      "message" -> "Function loaded successfully from Wolfram Function Repository"
    |>
  ]
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", load_code],
            capture_output=True,
            text=True,
            timeout=60,
        )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "function": function_name}, indent=2
        )


@mcp.tool()
async def search_data_repository(
    query: str,
    max_results: int = 10,
) -> str:
    """
    Search the Wolfram Data Repository for curated datasets.

    The Data Repository contains hundreds of curated, ready-to-use datasets
    covering science, economics, geography, and more.

    Args:
        query: Search query (e.g., "climate", "financial", "genomic")
        max_results: Maximum number of results (default 10)

    Returns:
        List of matching datasets with descriptions
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    search_code = f"""
Module[{{results}},
  results = Quiet[Check[
    Take[
      ResourceSearch[{{"ResourceType" -> "DataResource", "Name" -> "{query}"}}, "SnippetData"],
      UpTo[{max_results}]
    ],
    {{}}
  ]];
  
  If[results === {{}},
    results = Quiet[Check[
      Take[
        ResourceSearch[{{"ResourceType" -> "DataResource", "Keyword" -> "{query}"}}, "SnippetData"],
        UpTo[{max_results}]
      ],
      {{}}
    ]]
  ];
  
  <|
    "success" -> True,
    "query" -> "{query}",
    "count" -> Length[results],
    "datasets" -> Map[
      <|
        "name" -> #["Name"],
        "description" -> Quiet[Check[#["ShortDescription"], ""]]
      |> &,
      results
    ]
  |>
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", search_code],
            capture_output=True,
            text=True,
            timeout=60,
        )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "query": query}, indent=2)


@mcp.tool()
async def get_dataset_info(dataset_name: str) -> str:
    """
    Get detailed information about a Wolfram Data Repository dataset.

    Args:
        dataset_name: The name of the dataset

    Returns:
        Dataset metadata including description, size, and structure
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    info_code = f"""
Module[{{rd, info}},
  rd = Quiet[Check[
    ResourceObject["{dataset_name}"],
    $Failed
  ]];
  
  If[rd === $Failed,
    <|"success" -> False, "error" -> "Dataset not found"|>,
    <|
      "success" -> True,
      "name" -> "{dataset_name}",
      "description" -> Quiet[Check[rd["Description"], ""]],
      "content_types" -> Quiet[Check[rd["ContentTypes"], {{}}]],
      "documentation_link" -> Quiet[Check[rd["DocumentationLink"], ""]],
      "keywords" -> Quiet[Check[rd["Keywords"], {{}}]]
    |>
  ]
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", info_code],
            capture_output=True,
            text=True,
            timeout=30,
        )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "dataset": dataset_name}, indent=2
        )


@mcp.tool()
async def load_dataset(
    dataset_name: str,
    sample_size: Optional[int] = None,
) -> str:
    """
    Load a dataset from the Wolfram Data Repository.

    Args:
        dataset_name: The name of the dataset to load
        sample_size: Optional number of rows to return (for preview)

    Returns:
        Dataset information and sample data
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    sample_clause = f"Take[#, UpTo[{sample_size}]]&" if sample_size else "Identity"

    load_code = f"""
Module[{{data, info}},
  data = Quiet[Check[
    ResourceData["{dataset_name}"],
    $Failed
  ]];
  
  If[data === $Failed,
    <|"success" -> False, "error" -> "Failed to load dataset"|>,
    <|
      "success" -> True,
      "name" -> "{dataset_name}",
      "loaded" -> True,
      "type" -> Head[data],
      "dimensions" -> If[Head[data] === Dataset,
        Quiet[Check[Dimensions[Normal[data]], "Unknown"]],
        Quiet[Check[Dimensions[data], "Unknown"]]
      ],
      "sample" -> ToString[{sample_clause}[data], InputForm],
      "columns" -> If[Head[data] === Dataset,
        Quiet[Check[Keys[First[Normal[data]]], {{}}]],
        {{}}
      ]
    |>
  ]
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", load_code],
            capture_output=True,
            text=True,
            timeout=120,
        )

        raw_output = result.stdout.strip()
        parsed = _parse_wolfram_association(raw_output)

        return json.dumps(
            parsed if isinstance(parsed, dict) else {"raw": raw_output}, indent=2
        )

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "success": False,
                "error": "Dataset loading timed out - dataset may be large",
                "dataset": dataset_name,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "dataset": dataset_name}, indent=2
        )


# ============================================================================
# Long Computation Async Workflow
# ============================================================================

# In-memory job storage (would use persistent storage in production)
_computation_jobs: Dict[str, Dict[str, Any]] = {}


@mcp.tool()
async def submit_computation(
    code: str,
    name: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """
    Submit a long-running computation for background execution.

    Use this for computations that may take more than 60 seconds.
    Returns a job_id that can be used to check status and retrieve results.

    Args:
        code: Wolfram Language code to execute
        name: Optional descriptive name for the job
        timeout: Maximum execution time in seconds (default 300 = 5 minutes)

    Returns:
        Job ID and submission confirmation

    Examples:
        submit_computation("FactorInteger[2^256 - 1]", name="Large factorization")
        submit_computation("NIntegrate[Sin[x^2], {x, 0, 100}]")
    """
    import subprocess
    import shutil
    import uuid
    import threading
    import time

    job_id = str(uuid.uuid4())[:8]

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    _computation_jobs[job_id] = {
        "id": job_id,
        "name": name or f"Job {job_id}",
        "code": code,
        "status": "running",
        "submitted_at": time.time(),
        "timeout": timeout,
        "result": None,
        "error": None,
    }

    def run_computation():
        try:
            result = subprocess.run(
                [wolframscript, "-code", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                _computation_jobs[job_id]["status"] = "completed"
                _computation_jobs[job_id]["result"] = result.stdout.strip()
            else:
                _computation_jobs[job_id]["status"] = "failed"
                _computation_jobs[job_id]["error"] = result.stderr or "Execution failed"

        except subprocess.TimeoutExpired:
            _computation_jobs[job_id]["status"] = "timeout"
            _computation_jobs[job_id]["error"] = (
                f"Computation timed out after {timeout}s"
            )
        except Exception as e:
            _computation_jobs[job_id]["status"] = "failed"
            _computation_jobs[job_id]["error"] = str(e)

        _computation_jobs[job_id]["completed_at"] = time.time()

    thread = threading.Thread(target=run_computation, daemon=True)
    thread.start()

    return json.dumps(
        {
            "success": True,
            "job_id": job_id,
            "name": name or f"Job {job_id}",
            "status": "submitted",
            "message": f"Computation submitted. Use poll_computation('{job_id}') to check status.",
        },
        indent=2,
    )


@mcp.tool()
async def poll_computation(job_id: str) -> str:
    """
    Check the status of a submitted computation.

    Args:
        job_id: The job ID returned by submit_computation

    Returns:
        Current status (running, completed, failed, timeout)
    """
    import time

    if job_id not in _computation_jobs:
        return json.dumps(
            {"success": False, "error": f"Job '{job_id}' not found"}, indent=2
        )

    job = _computation_jobs[job_id]

    elapsed = time.time() - job["submitted_at"]

    return json.dumps(
        {
            "success": True,
            "job_id": job_id,
            "name": job["name"],
            "status": job["status"],
            "elapsed_seconds": round(elapsed, 1),
            "has_result": job["result"] is not None,
        },
        indent=2,
    )


@mcp.tool()
async def get_computation_result(job_id: str) -> str:
    """
    Retrieve the result of a completed computation.

    Args:
        job_id: The job ID returned by submit_computation

    Returns:
        The computation result if completed, or error information
    """
    if job_id not in _computation_jobs:
        return json.dumps(
            {"success": False, "error": f"Job '{job_id}' not found"}, indent=2
        )

    job = _computation_jobs[job_id]

    if job["status"] == "running":
        return json.dumps(
            {
                "success": False,
                "status": "running",
                "message": "Computation still in progress. Use poll_computation to check status.",
            },
            indent=2,
        )

    return json.dumps(
        {
            "success": job["status"] == "completed",
            "job_id": job_id,
            "name": job["name"],
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
        },
        indent=2,
    )


# ============================================================================
# Expression Caching
# ============================================================================


@mcp.tool()
async def cache_expression(name: str, expression: str) -> str:
    """
    Evaluate and cache a Wolfram expression for later reuse.

    Useful for expensive computations that may be needed multiple times.
    The result is stored in memory and can be retrieved by name.

    Args:
        name: Unique name to identify this cached expression
        expression: Wolfram Language expression to evaluate and cache

    Returns:
        The evaluation result and cache confirmation

    Examples:
        cache_expression("my_integral", "Integrate[Sin[x]^10, x]")
        cache_expression("primes", "Table[Prime[n], {n, 1, 1000}]")
    """
    if not FEATURES.expression_cache:
        return json.dumps(
            {"success": False, "error": "Expression caching is disabled"}, indent=2
        )

    result = await execute_code(code=expression, format="text", output_target="cli")

    try:
        result_data = json.loads(result)
        output = result_data.get("output", result)
    except (json.JSONDecodeError, TypeError):
        output = result

    success = _cache_expr(name, expression, str(output))

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
    """
    Retrieve a previously cached expression result.

    Args:
        name: The name used when caching the expression

    Returns:
        The cached result or error if not found
    """
    if not FEATURES.expression_cache:
        return json.dumps(
            {"success": False, "error": "Expression caching is disabled"}, indent=2
        )

    cached = _get_cached(name)

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
    """
    List all cached expressions with their metadata.

    Returns:
        Dictionary of cached expressions with access counts and ages
    """
    if not FEATURES.expression_cache:
        return json.dumps(
            {"success": False, "error": "Expression caching is disabled"}, indent=2
        )

    cached = list_cached_expressions()
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
    """
    Clear all cached expressions.

    Returns:
        Confirmation of cache clearance
    """
    if not FEATURES.expression_cache:
        return json.dumps(
            {"success": False, "error": "Expression caching is disabled"}, indent=2
        )

    clear_cache()
    return json.dumps(
        {"success": True, "message": "Expression cache cleared"}, indent=2
    )


# ============================================================================
# Feature Flags and Telemetry
# ============================================================================


@mcp.tool()
async def get_feature_status() -> str:
    """
    Get the status of all feature flags.

    Feature flags can be controlled via environment variables:
    - MATHEMATICA_ENABLE_FUNCTION_REPO: Wolfram Function Repository integration
    - MATHEMATICA_ENABLE_DATA_REPO: Wolfram Data Repository integration
    - MATHEMATICA_ENABLE_ASYNC: Async long computation workflow
    - MATHEMATICA_ENABLE_LOOKUP: Symbol lookup/introspection
    - MATHEMATICA_ENABLE_MATH_ALIASES: Named math operation shortcuts
    - MATHEMATICA_ENABLE_CACHE: Expression caching
    - MATHEMATICA_ENABLE_TELEMETRY: Usage telemetry (default: off)

    Returns:
        Current state of all feature flags
    """
    return json.dumps(
        {
            "success": True,
            "features": FEATURES.to_dict(),
        },
        indent=2,
    )


@mcp.tool()
async def get_telemetry_stats() -> str:
    """
    Get usage statistics for tools (if telemetry is enabled).

    Telemetry is disabled by default. Enable with MATHEMATICA_ENABLE_TELEMETRY=true.

    Returns:
        Tool usage statistics including call counts and timing
    """
    if not FEATURES.telemetry:
        return json.dumps(
            {
                "success": False,
                "error": "Telemetry is disabled. Set MATHEMATICA_ENABLE_TELEMETRY=true to enable.",
            },
            indent=2,
        )

    stats = get_usage_stats()
    return json.dumps(
        {
            "success": True,
            "tool_stats": stats,
        },
        indent=2,
    )


@mcp.tool()
async def reset_telemetry() -> str:
    """
    Reset all telemetry statistics to zero.

    Returns:
        Confirmation of reset
    """
    if not FEATURES.telemetry:
        return json.dumps(
            {"success": False, "error": "Telemetry is disabled"}, indent=2
        )

    reset_stats()
    return json.dumps(
        {"success": True, "message": "Telemetry statistics reset"}, indent=2
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
