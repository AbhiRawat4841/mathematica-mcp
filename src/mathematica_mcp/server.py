import json
import logging
import os
import re
import difflib
from typing import Literal, Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP, Image

from .connection import get_mathematica_connection, close_connection
from .session import (
    execute_in_kernel,
    get_kernel_session,
    close_kernel_session,
    _is_graphics_output,
    _rasterize_via_wolframscript,
    _session_context,
    _wrap_code_for_context,
    _wrap_code_for_determinism,
)
from .config import FEATURES
from .telemetry import telemetry_tool, get_usage_stats, reset_stats
from .cache import (
    cache_expression as _cache_expr,
    get_cached_expression as _get_cached,
    list_cached_expressions,
    clear_cache,
    remove_cached_expression,
)
from .error_analyzer import analyze_messages, format_error_for_llm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mathematica_mcp")

mcp = FastMCP("mathematica-mcp")


def _prepare_raster_code(
    code: str,
    *,
    deterministic_seed: Optional[int],
    session_id: Optional[str],
    isolate_context: bool,
) -> str:
    context = _session_context(session_id) if isolate_context else None
    wrapped = _wrap_code_for_context(code, context)
    return _wrap_code_for_determinism(wrapped, deterministic_seed)


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
        # Remove newlines within the association (multiline Mathematica output)
        s = re.sub(r'\n\s*', ' ', s)
        # Remove carriage returns
        s = s.replace('\r', '')
        # Handle fractions like 100584/625 - quote them as strings
        s = re.sub(r':\s*(\d+)/(\d+)\s*([,}])', r': "\1/\2"\3', s)
        # Convert Association delimiters
        s = re.sub(r"<\|", "{", s)
        s = re.sub(r"\|>", "}", s)
        # Convert arrow to colon
        s = re.sub(r"\s*->\s*", ": ", s)
        # Convert Mathematica booleans
        s = s.replace("True", "true").replace("False", "false").replace("Null", "null")
        # Quote unquoted symbols (identifiers starting with letter, may contain ` $ digits)
        # But be careful not to match already quoted strings or numbers
        s = re.sub(r':\s*([A-Za-z][A-Za-z0-9`$]*(?:\s+[A-Za-z][A-Za-z0-9`$]*)*)\s*([,}])', 
                   r': "\1"\2', s)
        # Handle special Mathematica output like "100584 kilometers" with line breaks
        s = re.sub(r':\s*(\d+)\s+([A-Za-z]+)\s*([,}])', r': "\1 \2"\3', s)
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
    try:
        conn = get_mathematica_connection()
        return conn.send_command(command, params or {})
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_mathematica_status() -> str:
    """Get connection status and system info."""
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
    """List all open Mathematica notebooks. Returns ID, filename, title."""
    result = _try_addon_command("get_notebooks")
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_notebook_info(
    notebook: Optional[str] = None, session_id: Optional[str] = None
) -> str:
    """Get details about a notebook (filename, directory, cell count)."""
    result = _try_addon_command(
        "get_notebook_info", {"notebook": notebook, "session_id": session_id}
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def create_notebook(title: str = "Untitled", session_id: Optional[str] = None) -> str:
    """Create a new empty notebook. Returns notebook ID.
    
    NOTE: For executing code in a notebook, prefer execute_code(code, output_target="notebook")
    which handles notebook creation, cell writing, and evaluation atomically.
    """
    result = _try_addon_command(
        "create_notebook", {"title": title, "session_id": session_id}
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def save_notebook(
    notebook: Optional[str] = None,
    path: Optional[str] = None,
    format: Literal["Notebook", "PDF", "HTML", "TeX"] = "Notebook",
    session_id: Optional[str] = None,
) -> str:
    """Save a notebook to disk."""
    result = _try_addon_command(
        "save_notebook",
        {
            "notebook": notebook,
            "path": path,
            "format": format,
            "session_id": session_id,
        },
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def close_notebook(
    notebook: Optional[str] = None, session_id: Optional[str] = None
) -> str:
    """Close a notebook."""
    result = _try_addon_command(
        "close_notebook", {"notebook": notebook, "session_id": session_id}
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_cells(
    notebook: Optional[str] = None,
    style: Optional[str] = None,
    session_id: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    include_content: bool = True,
) -> str:
    """Get list of cells in a notebook."""
    result = _try_addon_command(
        "get_cells",
        {
            "notebook": notebook,
            "style": style,
            "session_id": session_id,
            "offset": offset,
            "limit": limit,
            "include_content": include_content,
        },
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_cell_content(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Get the full content of a specific cell."""
    result = _try_addon_command(
        "get_cell_content",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def write_cell(
    content: str,
    style: str = "Input",
    notebook: Optional[str] = None,
    position: Literal["After", "Before", "End", "Beginning"] = "After",
    session_id: Optional[str] = None,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """Write a new cell to a notebook without evaluating it.
    
    NOTE: For executing code, prefer execute_code(code, output_target="notebook")
    which writes AND evaluates the cell atomically.
    """
    result = _try_addon_command(
        "write_cell",
        {
            "notebook": notebook,
            "content": content,
            "style": style,
            "position": position,
            "session_id": session_id,
            "sync": sync,
            "sync_wait": sync_wait,
        },
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Delete a cell from a notebook."""
    result = _try_addon_command(
        "delete_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def evaluate_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
    max_wait: int = 10,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """Evaluate a specific cell."""
    result = _try_addon_command(
        "evaluate_cell",
        {
            "cell_id": cell_id,
            "notebook": notebook,
            "session_id": session_id,
            "max_wait": max_wait,
            "sync": sync,
            "sync_wait": sync_wait,
        },
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def execute_code(
    code: str,
    format: Literal["text", "latex", "mathematica"] = "text",
    output_target: Literal["cli", "notebook"] = "notebook",
    mode: Literal["kernel", "frontend"] = "kernel",
    render_graphics: bool = True,
    deterministic_seed: Optional[int] = None,
    session_id: Optional[str] = None,
    isolate_context: bool = False,
    timeout: int = 60,
    max_wait: int = 30,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """Execute Wolfram Language code. THIS IS THE PRIMARY TOOL for running Mathematica code.

    Use this tool to evaluate integrals, solve equations, create plots, etc.
    With output_target="notebook", it atomically creates/finds a notebook, writes the code,
    and evaluates it - all in one fast operation.

    Args:
        code: Wolfram Language code (e.g., "Integrate[x^2, x]", "Plot[Sin[x], {x, 0, 2Pi}]")
        format: Output format (text, latex, mathematica)
        output_target: "notebook" (insert into active notebook) or "cli" (return text only)
        mode: "kernel" (fast, synchronous) or "frontend" (legacy, for Manipulate/dynamic content)
        render_graphics: Whether to auto-rasterize Graphics output to an image
        deterministic_seed: Seed for deterministic random output (optional)
        session_id: Optional session identifier for notebook routing and isolation
        isolate_context: Use a dedicated Mathematica context per session_id
        timeout: Max seconds for kernel evaluation (CLI paths)
        max_wait: Max seconds for notebook frontend evaluation
        sync: Notebook sync mode (none/refresh/strict)
        sync_wait: Seconds to wait in strict sync mode
    
    Examples:
        execute_code("Integrate[(2x+x^2)/x^5, x]") - Compute integral in notebook
        execute_code("Plot[Sin[x], {x, 0, 2Pi}]") - Create plot in notebook
        execute_code("Solve[x^2 - 4 == 0, x]", output_target="cli") - Get result as text
    """
    if output_target == "notebook":
        try:
            # Use atomic command that combines find/create notebook + write + evaluate
            # in a single round-trip for better performance
            # mode="kernel" is the new fast path (no polling)
            params = {
                "code": code,
                "max_wait": max_wait,
                "mode": mode,
                "session_id": session_id,
                "isolate_context": isolate_context,
                "sync": sync,
                "sync_wait": sync_wait,
            }
            if deterministic_seed is not None:
                params["deterministic_seed"] = deterministic_seed
            result = _try_addon_command("execute_code_notebook", params)

            if result.get("success"):
                response = {
                    "status": "executed_in_notebook",
                    "code": code,
                    "notebook_id": result.get("notebook_id"),
                    "cell_id": result.get("cell_id"),
                    "evaluated": True,
                    "message": "Executed in notebook.",
                }
                if result.get("created_notebook"):
                    response["note"] = "Created new notebook 'Analysis'."

                # NEW: Process error messages if present
                if result.get("has_errors") or result.get("has_warnings"):
                    messages = result.get("messages", [])
                    response["messages"] = messages
                    response["has_errors"] = result.get("has_errors", False)
                    response["has_warnings"] = result.get("has_warnings", False)

                    # Update status to indicate errors
                    if result.get("has_errors"):
                        response["status"] = "executed_with_errors"
                        response["message"] = (
                            "Code executed in notebook but produced errors. "
                            "See 'error_analysis' field for detailed suggestions."
                        )

                    # Format error summary for easy reading
                    error_msgs = [m for m in messages if m.get("type") == "error"]
                    warning_msgs = [m for m in messages if m.get("type") == "warning"]

                    if error_msgs or warning_msgs:
                        summary_parts = []
                        if error_msgs:
                            summary_parts.append(
                                f"{len(error_msgs)} error(s): "
                                + "; ".join(
                                    f"{m.get('tag', 'Unknown')}" for m in error_msgs[:3]
                                )
                            )
                        if warning_msgs:
                            summary_parts.append(
                                f"{len(warning_msgs)} warning(s): "
                                + "; ".join(
                                    f"{m.get('tag', 'Unknown')}"
                                    for m in warning_msgs[:3]
                                )
                            )
                        response["error_summary"] = " | ".join(summary_parts)

                        # NEW: Add intelligent error analysis
                        error_analysis = analyze_messages(messages)
                        response["error_analysis"] = {
                            "total_errors": error_analysis["errors"],
                            "total_warnings": error_analysis["warnings"],
                            "severity": error_analysis["severity"],
                            "recommendations": error_analysis["recommendations"],
                            "should_retry": error_analysis["should_retry"],
                        }

                        # Include detailed analysis for each error
                        if error_analysis.get("analyses"):
                            response["detailed_analyses"] = [
                                {
                                    "tag": a["original_message"]["tag"],
                                    "description": a.get("description", ""),
                                    "suggested_fix": a.get("suggested_fix", ""),
                                    "example": a.get("example", ""),
                                    "confidence": a.get("confidence", "low"),
                                }
                                for a in error_analysis["analyses"]
                                if a.get("confidence") in ["high", "medium"]
                            ]

                        # Add formatted error message for LLM
                        response["llm_error_report"] = format_error_for_llm(
                            messages, code
                        )

                        # Add output preview if available
                        if result.get("output_preview"):
                            response["output_preview"] = result.get("output_preview")

                return json.dumps(response, indent=2)
            else:
                raise RuntimeError(
                    result.get("error", "Atomic notebook execution failed")
                )

        except Exception as e:
            logger.warning(f"Notebook execution failed: {e}. Falling back to CLI.")

            # Fallback to CLI execution with auto-rasterization for graphics
            try:
                params = {
                    "code": code,
                    "format": format,
                    "session_id": session_id,
                    "isolate_context": isolate_context,
                    "timeout": timeout,
                }
                if deterministic_seed is not None:
                    params["deterministic_seed"] = deterministic_seed
                result = _try_addon_command("execute_code", params)
                if isinstance(result, dict):
                    result["note"] = "Executed via CLI (notebook error)."
                    output = result.get("output", "")
                    output_inputform = result.get("output_inputform", "")
                    if render_graphics and (
                        _is_graphics_output(output)
                        or _is_graphics_output(output_inputform)
                    ):
                        raster_code = _prepare_raster_code(
                            code,
                            deterministic_seed=deterministic_seed,
                            session_id=session_id,
                            isolate_context=isolate_context,
                        )
                        image_path = _rasterize_via_wolframscript(raster_code)
                        if image_path:
                            result["rendered_image"] = image_path
                            result["output"] = (
                                f"[Graphics rendered to image: {image_path}]"
                            )
                            result["is_graphics"] = True
                            result["tip"] = "Use Read tool to view image."
                    return json.dumps(result, indent=2)
                return f"{result}\n(Note: Executed via CLI due to notebook error)"
            except Exception:
                result = execute_in_kernel(
                    code,
                    format,
                    render_graphics=render_graphics,
                    deterministic_seed=deterministic_seed,
                    session_id=session_id,
                    isolate_context=isolate_context,
                    timeout=timeout,
                )
                result["execution_mode"] = "kernel_fallback"
                result["note"] = (
                    "Executed via CLI (notebook error). Run StartMCPServer[] in Mathematica."
                )
                # Return image info if graphics were rasterized
                if result.get("is_graphics") and result.get("image_path"):
                    result["rendered_image"] = result["image_path"]
                    result["tip"] = "Use Read tool to view image."
                return json.dumps(result, indent=2)

    params = {
        "code": code,
        "format": format,
        "session_id": session_id,
        "isolate_context": isolate_context,
        "timeout": timeout,
    }
    if deterministic_seed is not None:
        params["deterministic_seed"] = deterministic_seed
    result = _try_addon_command("execute_code", params)
    # Check if addon command succeeded, otherwise fall back to kernel
    if result.get("success") is False or "error" in result:
        result = execute_in_kernel(
            code,
            format,
            render_graphics=render_graphics,
            deterministic_seed=deterministic_seed,
            session_id=session_id,
            isolate_context=isolate_context,
            timeout=timeout,
        )
        result["execution_mode"] = "kernel_fallback"
        # Return image info if graphics were rasterized
        if result.get("is_graphics") and result.get("image_path"):
            result["rendered_image"] = result["image_path"]
            result["tip"] = "Use Read tool to view image."
    else:
        # Check if addon returned Graphics output and rasterize it
        output = result.get("output", "")
        output_inputform = result.get("output_inputform", "")
        if render_graphics and (
            _is_graphics_output(output) or _is_graphics_output(output_inputform)
        ):
            raster_code = _prepare_raster_code(
                code,
                deterministic_seed=deterministic_seed,
                session_id=session_id,
                isolate_context=isolate_context,
            )
            image_path = _rasterize_via_wolframscript(raster_code)
            if image_path:
                result["rendered_image"] = image_path
                result["output"] = f"[Graphics rendered to image: {image_path}]"
                result["is_graphics"] = True
                result["tip"] = "Use Read tool to view image."
                # Keep a short preview of the raw output for debugging
                if output_inputform:
                    result["output_inputform"] = (
                        output_inputform[:200] + "..."
                        if len(output_inputform) > 200
                        else output_inputform
                    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def batch_commands(commands: List[Dict[str, Any]]) -> str:
    """Execute multiple commands in one round-trip."""
    result = _try_addon_command("batch_commands", {"commands": commands})
    return json.dumps(result, indent=2)


@mcp.tool()
async def evaluate_selection(
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
    max_wait: int = 10,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """
    Evaluate the currently selected cell(s) in a notebook.

    Args:
        notebook: Notebook ID. If None, uses selected notebook.
    """
    result = _try_addon_command(
        "execute_selection",
        {
            "notebook": notebook,
            "session_id": session_id,
            "max_wait": max_wait,
            "sync": sync,
            "sync_wait": sync_wait,
        },
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def screenshot_notebook(
    notebook: Optional[str] = None,
    max_height: int = 2000,
    session_id: Optional[str] = None,
    use_rasterize: bool = False,
    wait_ms: int = 100,
) -> Image:
    """
    Capture a screenshot of an entire notebook window.

    Args:
        notebook: Notebook ID. If None, uses selected notebook.
        max_height: Maximum height in pixels (prevents huge images)

    Returns the screenshot as an image that can be viewed directly.
    """
    result = _try_addon_command(
        "screenshot_notebook",
        {
            "notebook": notebook,
            "max_height": max_height,
            "session_id": session_id,
            "use_rasterize": use_rasterize,
            "wait_ms": wait_ms,
        },
    )

    image_path = result["path"]
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    os.remove(image_path)
    return Image(data=image_bytes, format="png")


@mcp.tool()
async def screenshot_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
    use_rasterize: bool = False,
) -> Image:
    """
    Capture a screenshot of a specific cell's content and output.

    Useful for seeing plots, graphics, or formatted mathematical output.

    Args:
        cell_id: The cell object ID to screenshot
    """
    result = _try_addon_command(
        "screenshot_cell",
        {
            "cell_id": cell_id,
            "notebook": notebook,
            "session_id": session_id,
            "use_rasterize": use_rasterize,
        },
    )

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
async def select_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Select a cell in the notebook (moves cursor to it)."""
    result = _try_addon_command(
        "select_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def scroll_to_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Scroll the notebook view to make a cell visible."""
    result = _try_addon_command(
        "scroll_to_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def export_notebook(
    path: str,
    notebook: Optional[str] = None,
    format: Literal["PDF", "HTML", "TeX", "Markdown"] = "PDF",
    session_id: Optional[str] = None,
) -> str:
    """Export a notebook to PDF, HTML, TeX, or Markdown."""
    result = _try_addon_command(
        "export_notebook",
        {"notebook": notebook, "path": path, "format": format, "session_id": session_id},
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
    """Search for Wolfram Language functions and optionally auto-execute."""
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
    """Verify a sequence of mathematical expressions steps."""
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
    """Get comprehensive information about a Wolfram Language symbol."""
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
    """Get current Wolfram kernel session state (memory, uptime, version)."""
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
    """Load a Mathematica package (e.g., "Developer`")."""
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
    """List all currently loaded packages and contexts."""
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
    """Compute integral using Integrate."""
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
    """Solve an equation using Solve."""
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
    """Simplify a mathematical expression."""
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
    """Compute derivative using D."""
    if order == 1:
        code = f"D[{expression}, {variable}]"
    else:
        code = f"D[{expression}, {{{variable}, {order}}}]"

    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_expand(expression: str) -> str:
    """Expand a mathematical expression."""
    code = f"Expand[{expression}]"
    result = await execute_code(code=code, format="text", output_target="cli")
    return result


@mcp.tool()
async def mathematica_factor(expression: str) -> str:
    """Factor a mathematical expression."""
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
    """Compute limit using Limit."""
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
    """Compute Taylor/power series expansion."""
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
    """Search the Wolfram Function Repository."""
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    safe_query = query.replace('"', '\\"')

    search_code = f"""
Module[{{results, query, clean, fetch, maxRes}},
  query = "{safe_query}";
  maxRes = {max_results};
  fetch[q_, field_] := Quiet[Check[
    Normal@ResourceSearch[{{"ResourceType" -> "Function", field -> q}}, "Associations"],
    {{}}
  ]];

  results = Quiet[Check[Take[fetch[query, "Name"], UpTo[maxRes]], {{}}]];

  If[results === {{}},
    results = Quiet[Check[Take[fetch[query, "Keyword"], UpTo[maxRes]], {{}}]]
  ];

  clean[res_] := <|
    "name" -> ToString[Lookup[res, "Name", ""]],
    "short_description" -> ToString[Lookup[res, "ShortDescription", ""]],
    "repository_location" -> "Wolfram Function Repository"
  |>;

  ExportString[
    <|
      "success" -> True,
      "query" -> query,
      "count" -> Length[results],
      "results" -> Map[clean, results]
    |>,
    "JSON"
  ]
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", search_code],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return json.dumps(
                {
                    "success": False,
                    "error": result.stderr or "Search failed",
                    "query": query,
                },
                indent=2,
            )

        raw_output = result.stdout.strip()
        if not raw_output:
            return json.dumps(
                {"success": False, "error": "Empty search response", "query": query},
                indent=2,
            )

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Failed to parse search response: {e}",
                    "raw": raw_output,
                },
                indent=2,
            )

        return json.dumps(parsed, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps(
            {"success": False, "error": "Search timed out", "query": query}, indent=2
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "query": query}, indent=2)


@mcp.tool()
async def get_function_repository_info(function_name: str) -> str:
    """Get details about a Wolfram Function Repository function."""
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
    """Load a function from the Wolfram Function Repository."""
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
    """Search the Wolfram Data Repository."""
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
    """Get detailed information about a Wolfram Data Repository dataset."""
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
    """Load a dataset from the Wolfram Data Repository."""
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
import threading as _threading

_computation_jobs_lock = _threading.Lock()
_MAX_JOBS = 100
_MAX_JOB_AGE_SECONDS = 3600


@mcp.tool()
async def submit_computation(
    code: str,
    name: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Submit a long-running computation for background execution."""
    import subprocess
    import shutil
    import uuid
    import threading
    import time

    def _prune_jobs(now: float) -> None:
        expired = [
            job_id
            for job_id, job in _computation_jobs.items()
            if now - job.get("submitted_at", now) > _MAX_JOB_AGE_SECONDS
        ]
        for job_id in expired:
            _computation_jobs.pop(job_id, None)

    job_id = str(uuid.uuid4())[:8]

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found in PATH"}, indent=2
        )

    now = time.time()
    with _computation_jobs_lock:
        _prune_jobs(now)
        if len(_computation_jobs) >= _MAX_JOBS:
            return json.dumps(
                {
                    "success": False,
                    "error": "Too many active jobs. Try again later.",
                },
                indent=2,
            )
        _computation_jobs[job_id] = {
            "id": job_id,
            "name": name or f"Job {job_id}",
            "code": code,
            "status": "running",
            "submitted_at": now,
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

            with _computation_jobs_lock:
                if result.returncode == 0:
                    _computation_jobs[job_id]["status"] = "completed"
                    _computation_jobs[job_id]["result"] = result.stdout.strip()
                else:
                    _computation_jobs[job_id]["status"] = "failed"
                    _computation_jobs[job_id]["error"] = (
                        result.stderr or "Execution failed"
                    )

        except subprocess.TimeoutExpired:
            with _computation_jobs_lock:
                _computation_jobs[job_id]["status"] = "timeout"
                _computation_jobs[job_id]["error"] = (
                    f"Computation timed out after {timeout}s"
                )
        except Exception as e:
            with _computation_jobs_lock:
                _computation_jobs[job_id]["status"] = "failed"
                _computation_jobs[job_id]["error"] = str(e)

        with _computation_jobs_lock:
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
    """Check the status of a submitted computation."""
    import time

    with _computation_jobs_lock:
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
    """Retrieve the result of a completed computation."""
    with _computation_jobs_lock:
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
    """Evaluate and cache a Wolfram expression for later reuse."""
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
    """Retrieve a previously cached expression result."""
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
    """List all cached expressions with their metadata."""
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
    """Clear all cached expressions."""
    if not FEATURES.expression_cache:
        return json.dumps(
            {"success": False, "error": "Expression caching is disabled"}, indent=2
        )

    clear_cache()
    return json.dumps(
        {"success": True, "message": "Expression cache cleared"}, indent=2
    )


# ============================================================================
# TIER 1: Variable Introspection (Session State)
# ============================================================================


@mcp.tool()
async def list_variables(include_system: bool = False) -> str:
    """List all user-defined variables in the current Mathematica kernel session."""
    result = _try_addon_command("list_variables", {"include_system": include_system})

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_variable(name: str) -> str:
    """Get detailed information about a specific variable."""
    result = _try_addon_command("get_variable", {"name": name})

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def set_variable(name: str, value: str) -> str:
    """
    Set a variable in the Mathematica kernel session.

    Args:
        name: Variable name (e.g., "x", "myData")
        value: Wolfram Language expression to assign (e.g., "5", "{1,2,3}", "Plot[Sin[x],{x,0,Pi}]")

    Returns:
        Confirmation with the assigned value

    Example:
        set_variable("x", "Range[10]") -> {success: true, value: "{1,2,3,4,5,6,7,8,9,10}"}
    """
    result = _try_addon_command("set_variable", {"name": name, "value": value})

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def clear_variables(
    names: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    clear_all: bool = False,
) -> str:
    """
    Clear variables from the Mathematica kernel session.

    Equivalent to Python's 'del' or clearing notebook state.

    Args:
        names: Specific variable names to clear (e.g., ["x", "y", "z"])
        pattern: Wolfram pattern to match (e.g., "temp*" clears temp1, temp2, etc.)
        clear_all: If True, clear ALL Global` variables (use with caution!)

    Returns:
        List of cleared variables

    Example:
        clear_variables(names=["x", "y"]) -> {cleared: ["x", "y"], count: 2}
        clear_variables(pattern="temp*") -> {cleared: ["temp1", "temp2"], count: 2}
    """
    params = {}
    if names:
        params["names"] = names
    if pattern:
        params["pattern"] = pattern
    if clear_all:
        params["clear_all"] = True

    result = _try_addon_command("clear_variables", params)

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_expression_info(expression: str) -> str:
    """
    Get detailed structural information about a Wolfram expression.

    Like Python's type() on steroids - shows Head, FullForm, tree structure,
    depth, leaf count, and type checks (NumericQ, ListQ, etc.)

    Args:
        expression: Wolfram Language expression to analyze

    Returns:
        Structural information: head, full form, depth, leaf count, type flags

    Example:
        get_expression_info("{{1,2},{3,4}}") -> {head: "List", depth: 3, dimensions: [2,2]}
        get_expression_info("Sin[x] + Cos[x]") -> {head: "Plus", leaf_count: 3}
    """
    result = _try_addon_command("get_expression_info", {"expression": expression})

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


# ============================================================================
# TIER 1: Error Recovery
# ============================================================================


@mcp.tool()
async def get_messages(count: int = 10) -> str:
    """
    Get recent Mathematica messages/warnings from the session.

    Like Python's exception traceback - helps debug what went wrong.

    Args:
        count: Number of recent messages to retrieve (default 10)

    Returns:
        List of recent messages with timestamps

    Example:
        After a failed computation:
        get_messages() -> [{timestamp: "...", message: "Power::infy: Infinite expression 1/0 encountered."}]
    """
    result = _try_addon_command("get_messages", {"count": count})

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def restart_kernel() -> str:
    """
    Restart the Mathematica kernel, clearing all state.

    This is the nuclear option - clears all variables, definitions, and state.
    Use when the kernel is in a bad state or you need a fresh start.

    Returns:
        Confirmation of kernel restart
    """
    close_kernel_session()
    # Force reconnection
    result = _try_addon_command("ping")

    return json.dumps(
        {
            "success": True,
            "message": "Kernel session cleared. Fresh session will be created on next execution.",
            "ping_result": result,
        },
        indent=2,
    )


# ============================================================================
# TIER 2: File Handling (.nb, .wl, .wlnb)
# ============================================================================


def _expand_path(path: str) -> str:
    """Expand ~ and make path absolute."""
    expanded = os.path.expanduser(path)
    return os.path.abspath(expanded)


@mcp.tool()
async def open_notebook_file(
    path: str, session_id: Optional[str] = None
) -> str:
    """
    Open an existing Mathematica notebook file (.nb) in the Mathematica frontend.

    Supports:
    - Absolute paths: /Users/foo/notebook.nb
    - Home-relative paths: ~/Documents/notebook.nb
    - Relative paths (resolved from current directory)

    Args:
        path: Path to the .nb file

    Returns:
        Notebook ID and metadata for use with other notebook commands

    Example:
        open_notebook_file("~/Documents/analysis.nb") -> {id: "NotebookObject[...]", cell_count: 15}
    """
    expanded = _expand_path(path)
    result = _try_addon_command(
        "open_notebook_file", {"path": expanded, "session_id": session_id}
    )

    if result.get("error"):
        return json.dumps(
            {"success": False, "error": result["error"], "path": expanded}, indent=2
        )

    return json.dumps(result, indent=2)


@mcp.tool()
async def run_script(path: str) -> str:
    """
    Execute a Wolfram Language script file (.wl, .m) and return the result.

    This is equivalent to Get[path] - loads and executes the script in the
    current kernel session. Any definitions or side effects persist.

    Args:
        path: Path to the .wl or .m script file

    Returns:
        The result of the last expression in the script, plus timing info

    Example:
        run_script("~/scripts/setup.wl") -> {result: "Null", timing_ms: 150}
    """
    expanded = _expand_path(path)
    result = _try_addon_command("run_script", {"path": expanded})

    if result.get("error"):
        return json.dumps(
            {"success": False, "error": result["error"], "path": expanded}, indent=2
        )

    return json.dumps(result, indent=2)


@mcp.tool()
async def read_notebook_content(path: str, include_outputs: bool = False) -> str:
    """
    Read the content of a notebook file as structured text.

    Extracts all cells from a notebook file without opening it in the frontend.
    Useful for understanding what's in a notebook before opening it.

    Args:
        path: Path to the .nb file
        include_outputs: If True, include Output cells (default: only Input/Text)

    Returns:
        Structured list of cells with their content and styles
    """
    import subprocess
    import shutil

    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    def _parse_with_python_parser(expanded_path: str) -> str:
        from .notebook_parser import NotebookParser, CellStyle

        parser = NotebookParser(truncation_threshold=25000)
        notebook = parser.parse_file(expanded_path)

        allowed_styles = {
            CellStyle.INPUT,
            CellStyle.CODE,
            CellStyle.TEXT,
            CellStyle.SECTION,
            CellStyle.SUBSECTION,
            CellStyle.TITLE,
        }

        cells = []
        for cell in notebook.cells:
            if include_outputs or cell.style in allowed_styles:
                cells.append({"content": cell.content, "style": cell.style.value})

        return json.dumps(
            {
                "success": True,
                "path": expanded_path,
                "cell_count": len(cells),
                "cells": cells[:50],
            },
            indent=2,
        )

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        try:
            return _parse_with_python_parser(expanded)
        except Exception as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"wolframscript not found and python parser failed: {e}",
                },
                indent=2,
            )

    style_filter = (
        "True"
        if include_outputs
        else 'MemberQ[{"Input", "Code", "Text", "Section", "Subsection", "Title"}, #["style"]]'
    )

    safe_path = expanded.replace("\\", "\\\\").replace('"', '\\"')

    code = f'''
Module[{{nb, cells, filtered}},
  nb = Quiet[Check[Import["{safe_path}", "Notebook"], $Failed]];
  If[nb === $Failed,
    ExportString[<|"success" -> False, "error" -> "Failed to read notebook"|>, "JSON"],
    cells = Cases[nb, Cell[content_, style_, ___] :> <|
      "content" -> ToString[content, InputForm],
      "style" -> ToString[style]
    |>, Infinity];
    filtered = Select[cells, {style_filter} &];
    ExportString[<|
      "success" -> True,
      "path" -> "{safe_path}",
      "cell_count" -> Length[filtered],
      "cells" -> Take[filtered, UpTo[50]]
    |>, "JSON"]
  ]
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "wolframscript failed")

        output = result.stdout.strip()
        if not output:
            raise RuntimeError("Empty response from wolframscript")

        try:
            return json.dumps(json.loads(output), indent=2)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse wolframscript JSON: {e}")
    except Exception as e:
        try:
            return _parse_with_python_parser(expanded)
        except Exception as parser_error:
            return json.dumps(
                {
                    "success": False,
                    "error": f"{e}; fallback parser failed: {parser_error}",
                },
                indent=2,
            )


@mcp.tool()
async def convert_notebook(
    path: str,
    output_format: Literal["markdown", "latex", "plain", "wolfram"] = "markdown",
) -> str:
    """
    Convert a Mathematica notebook to another format.

    Supported formats:
    - markdown: Readable Markdown with code blocks
    - latex: LaTeX document
    - plain: Plain text
    - wolfram: Pure Wolfram Language code only

    Args:
        path: Path to the .nb file
        output_format: Target format

    Returns:
        Converted content as a string
    """
    import subprocess
    import shutil

    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    format_map = {
        "markdown": '"Text"',
        "latex": '"TeX"',
        "plain": '"Text"',
        "wolfram": '"Text"',
    }

    code = f'''
Module[{{nb, content}},
  nb = Import["{expanded}"];
  content = ExportString[nb, {format_map.get(output_format, '"Text"')}];
  <|"success" -> True, "format" -> "{output_format}", "content" -> content|>
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def get_notebook_outline(path: str) -> str:
    """
    Get the structural outline of a notebook (sections, subsections, titles).

    Like a table of contents - shows the organization without full content.

    Args:
        path: Path to the .nb file

    Returns:
        Hierarchical outline of notebook sections
    """
    import subprocess
    import shutil

    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    code = f'''
Module[{{nb, sections}},
  nb = Quiet[Check[Import["{expanded}", "Notebook"], $Failed]];
  If[nb === $Failed,
    <|"success" -> False, "error" -> "Failed to read notebook"|>,
    sections = Cases[nb, 
      Cell[content_, style:"Title"|"Section"|"Subsection"|"Subsubsection"|"Chapter", ___] :> <|
        "level" -> style,
        "title" -> If[StringQ[content], content, ToString[content]]
      |>, 
      Infinity
    ];
    <|
      "success" -> True,
      "path" -> "{expanded}",
      "sections" -> sections,
      "count" -> Length[sections]
    |>
  ]
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def parse_notebook_python(
    path: str,
    output_format: Literal["markdown", "wolfram", "outline", "json"] = "markdown",
    truncation_threshold: int = 25000,
) -> str:
    """
    Parse a Mathematica notebook using Python-native parser.

    This tool provides offline notebook parsing without requiring wolframscript.
    It extracts clean, readable Wolfram code from complex BoxData structures.

    Args:
        path: Path to the .nb file
        output_format:
            - "markdown": Readable Markdown with code blocks (default)
            - "wolfram": Pure executable Wolfram Language code only
            - "outline": Hierarchical section outline
            - "json": Structured JSON with all cell data
        truncation_threshold: Max chars per cell before truncation (default 25000).
            Set to 0 to disable truncation (may timeout on large notebooks).

    Returns:
        Notebook content in the requested format
    """
    from .notebook_parser import NotebookParser

    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    try:
        effective_threshold = (
            truncation_threshold if truncation_threshold > 0 else float("inf")
        )
        parser = NotebookParser(
            truncation_threshold=int(effective_threshold)
            if effective_threshold != float("inf")
            else 10**9
        )
        notebook = parser.parse_file(expanded)

        if output_format == "markdown":
            content = parser.to_markdown(notebook)
            return json.dumps(
                {
                    "success": True,
                    "format": "markdown",
                    "path": expanded,
                    "cell_count": len(notebook.cells),
                    "code_cells": len(notebook.get_code_cells()),
                    "content": content,
                },
                indent=2,
            )

        elif output_format == "wolfram":
            content = parser.to_wolfram_code(notebook)
            return json.dumps(
                {
                    "success": True,
                    "format": "wolfram",
                    "path": expanded,
                    "code_cells": len(notebook.get_code_cells()),
                    "content": content,
                },
                indent=2,
            )

        elif output_format == "outline":
            outline = notebook.get_outline()
            return json.dumps(
                {
                    "success": True,
                    "format": "outline",
                    "path": expanded,
                    "section_count": len(outline),
                    "sections": outline,
                },
                indent=2,
            )

        elif output_format == "json":
            cells_data = [
                {
                    "index": c.cell_index,
                    "style": c.style.value,
                    "label": c.cell_label,
                    "content": c.content[:500] if len(c.content) > 500 else c.content,
                    "content_length": len(c.content),
                    "truncated_in_json": len(c.content) > 500,
                    "was_truncated": c.was_truncated,
                    "original_length": c.original_length
                    if c.was_truncated
                    else len(c.content),
                }
                for c in notebook.cells
            ]
            return json.dumps(
                {
                    "success": True,
                    "format": "json",
                    "path": expanded,
                    "title": notebook.title,
                    "cell_count": len(notebook.cells),
                    "code_cells": len(notebook.get_code_cells()),
                    "cells": cells_data,
                },
                indent=2,
            )

        else:
            return json.dumps(
                {"success": False, "error": f"Unknown format: {output_format}"},
                indent=2,
            )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def get_notebook_cell(
    path: str,
    cell_index: int,
    full: bool = False,
) -> str:
    """
    Get the full content of a specific cell by index.

    Use parse_notebook_python with format="json" to see all cell indices first.

    Args:
        path: Path to the .nb file
        cell_index: Cell index (0-based)
        full: If True, bypass truncation and return complete cell content (may be very large)

    Returns:
        Full cell content and metadata
    """
    from .notebook_parser import NotebookParser

    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    try:
        threshold = 10**9 if full else 25000
        parser = NotebookParser(truncation_threshold=threshold)
        notebook = parser.parse_file(expanded)

        if cell_index < 0 or cell_index >= len(notebook.cells):
            return json.dumps(
                {
                    "success": False,
                    "error": f"Cell index {cell_index} out of range (0-{len(notebook.cells) - 1})",
                },
                indent=2,
            )

        cell = notebook.cells[cell_index]
        return json.dumps(
            {
                "success": True,
                "index": cell.cell_index,
                "style": cell.style.value,
                "label": cell.cell_label,
                "content": cell.content,
                "content_length": len(cell.content),
                "was_truncated": cell.was_truncated,
                "original_length": cell.original_length
                if cell.was_truncated
                else len(cell.content),
                "raw_content_preview": cell.raw_content[:500]
                if cell.raw_content
                else "",
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# ============================================================================
# TIER 3: Wolfram Alpha & Natural Language
# ============================================================================


@mcp.tool()
async def wolfram_alpha(
    query: str,
    return_type: Literal["result", "data", "full"] = "result",
) -> str:
    """
    Query Wolfram Alpha with natural language.

    This gives Mathematica superpowers - ask questions in plain English
    and get computed answers, data, and more.

    Args:
        query: Natural language question (e.g., "population of France",
               "integrate x^2 from 0 to 1", "weather in Tokyo")
        return_type:
            - "result": Simple text result (default)
            - "data": Structured data when available
            - "full": All available pods/information

    Returns:
        Wolfram Alpha response in requested format

    Example:
        wolfram_alpha("population of Tokyo") -> "13.96 million people (2021)"
        wolfram_alpha("derivative of sin(x^2)", "data") -> {result: "2 x cos(x^2)"}
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    # Escape quotes in query
    safe_query = query.replace('"', '\\"')

    if return_type == "full":
        code = f'''
Module[{{result}},
  result = Quiet[Check[WolframAlpha["{safe_query}", "FullOutput"], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> "{safe_query}", "result" -> ToString[result, InputForm]|>
  ]
]
'''
    elif return_type == "data":
        code = f'''
Module[{{result}},
  result = Quiet[Check[WolframAlpha["{safe_query}", {{"Result", "Input"}}], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> "{safe_query}", 
      "result" -> ToString[result, InputForm]|>
  ]
]
'''
    else:
        code = f'''
Module[{{result}},
  result = Quiet[Check[WolframAlpha["{safe_query}", "Result"], $Failed]];
  If[result === $Failed,
    <|"success" -> False, "error" -> "Query failed"|>,
    <|"success" -> True, "query" -> "{safe_query}", "result" -> ToString[result]|>
  ]
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Query timed out"}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def interpret_natural_language(text: str) -> str:
    """
    Convert natural language mathematical description to Wolfram Language code.

    This is magic - describe what you want in English and get executable code.

    Args:
        text: Natural language description (e.g., "the integral of x squared from 0 to 1",
              "solve x squared equals 4 for x", "plot sine of x from 0 to 2 pi")

    Returns:
        Wolfram Language code and its evaluation result

    Example:
        interpret_natural_language("the derivative of e to the x")
        -> {code: "D[E^x, x]", result: "E^x"}
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    safe_text = text.replace('"', '\\"')

    code = '''
Module[{query, props, inputSpec, inputExpr, inputForm, resultExpr},
  query = "__MCP_QUERY__";
  props = Quiet[Check[WolframAlpha[query, "Properties"], {}]];
  inputSpec = SelectFirst[
    props,
    (Length[#] == 2 && #[[2]] === "Input") &,
    Missing["NotAvailable"]
  ];
  inputExpr = If[inputSpec === Missing["NotAvailable"],
    Missing["NotAvailable"],
    Quiet[Check[WolframAlpha[query, {inputSpec[[1]], "Input"}], Missing["NotAvailable"]]]
  ];

  inputForm = Which[
    inputExpr === Missing["NotAvailable"] || inputExpr === {}, "",
    Head[inputExpr] === HoldComplete,
      ToString[Unevaluated[inputExpr /. HoldComplete[x_] :> x], InputForm],
    True,
      ToString[Unevaluated[inputExpr], InputForm]
  ];

  resultExpr = Quiet[Check[WolframAlpha[query, "Result"], $Failed]];
  If[resultExpr === $Failed || resultExpr === Missing["NotAvailable"] || resultExpr === {},
    ExportString[<|"success" -> False, "error" -> "Could not interpret text"|>, "JSON"],
    ExportString[
      <|
        "success" -> True,
        "input" -> query,
        "wolfram_code" -> inputForm,
        "result" -> ToString[resultExpr, InputForm],
        "tex" -> Quiet[Check[ToString[TeXForm[resultExpr]], ""]]
      |>,
      "JSON"
    ]
  ]
]
'''
    code = code.replace("__MCP_QUERY__", safe_text)

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return json.dumps(
                {"success": False, "error": result.stderr or "Query failed"},
                indent=2,
            )

        output = result.stdout.strip()
        if not output:
            return json.dumps(
                {"success": False, "error": "Empty WolframAlpha response"},
                indent=2,
            )

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Failed to parse WolframAlpha response: {e}",
                    "raw": output,
                },
                indent=2,
            )

        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def entity_lookup(
    entity_type: str,
    name: str,
    properties: Optional[List[str]] = None,
) -> str:
    """
    Look up real-world entity data from Wolfram's curated knowledge base.

    Entity types include: "Country", "City", "Chemical", "Planet", "Company",
    "Person", "Movie", "University", "Element", "Star", and many more.

    Args:
        entity_type: Type of entity (e.g., "Country", "City", "Chemical")
        name: Name to look up (e.g., "France", "Tokyo", "Water")
        properties: Specific properties to retrieve (default: common properties)

    Returns:
        Entity data with requested properties

    Example:
        entity_lookup("Country", "Japan", ["Population", "Capital", "GDP"])
        -> {name: "Japan", Population: "125.8 million", Capital: "Tokyo", GDP: "$4.94 trillion"}
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    safe_name = name.replace('"', '\\"')

    if properties:
        props_str = "{" + ", ".join(f'"{p}"' for p in properties) + "}"
        code = f'''
Module[{{entity, data}},
  entity = Quiet[Check[Entity["{entity_type}", "{safe_name}"], $Failed]];
  If[entity === $Failed || !EntityQ[entity],
    <|"success" -> False, "error" -> "Entity not found"|>,
    data = EntityValue[entity, {props_str}];
    <|
      "success" -> True,
      "entity_type" -> "{entity_type}",
      "name" -> "{safe_name}",
      "properties" -> Map[ToString, data]
    |>
  ]
]
'''
    else:
        code = f'''
Module[{{entity, props, data}},
  entity = Quiet[Check[Entity["{entity_type}", "{safe_name}"], $Failed]];
  If[entity === $Failed || !EntityQ[entity],
    <|"success" -> False, "error" -> "Entity not found"|>,
    props = Take[EntityProperties[entity], UpTo[10]];
    data = EntityValue[entity, props];
    <|
      "success" -> True,
      "entity_type" -> "{entity_type}",
      "name" -> EntityValue[entity, "Name"],
      "properties" -> MapThread[Rule, {{props, Map[ToString, data]}}]
    |>
  ]
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def convert_units(quantity: str, target_unit: str) -> str:
    """
    Convert between units using Wolfram's comprehensive unit system.

    Args:
        quantity: Value with unit (e.g., "5 miles", "100 kg", "25 Celsius")
        target_unit: Target unit (e.g., "kilometers", "pounds", "Fahrenheit")

    Returns:
        Converted quantity

    Example:
        convert_units("100 kilometers", "miles") -> "62.1371 miles"
        convert_units("0 Celsius", "Fahrenheit") -> "32 Fahrenheit"
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    safe_qty = quantity.replace('"', '\\"')
    safe_unit = target_unit.replace('"', '\\"')

    code = f'''
Module[{{input, result}},
  input = Quiet[Check[Interpreter["Quantity"]["{safe_qty}"], $Failed]];
  If[input === $Failed,
    <|"success" -> False, "error" -> "Could not parse quantity"|>,
    result = Quiet[Check[UnitConvert[input, "{safe_unit}"], $Failed]];
    If[result === $Failed,
      <|"success" -> False, "error" -> "Conversion failed"|>,
      <|
        "success" -> True,
        "input" -> "{safe_qty}",
        "target_unit" -> "{safe_unit}",
        "result" -> ToString[result],
        "numeric" -> ToString[QuantityMagnitude[result]]
      |>
    ]
  ]
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def get_constant(name: str) -> str:
    """
    Get a physical or mathematical constant.

    Args:
        name: Constant name (e.g., "SpeedOfLight", "PlanckConstant", "Pi",
              "EulerGamma", "GoldenRatio", "Avogadro")

    Returns:
        Constant value with unit (if applicable) and numeric approximation

    Example:
        get_constant("SpeedOfLight") -> {value: "299792458 m/s", numeric: "2.998e8"}
    """
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    code = f'''
Module[{{constant, val, numeric}},
  constant = Quiet[Check[ToExpression["{name}"], $Failed]];
  If[constant === $Failed,
    constant = Quiet[Check[Quantity["{name}"], $Failed]]
  ];
  If[constant === $Failed,
    <|"success" -> False, "error" -> "Constant not found"|>,
    <|
      "success" -> True,
      "name" -> "{name}",
      "exact" -> ToString[constant, InputForm],
      "numeric" -> ToString[N[constant, 15]],
      "tex" -> Quiet[Check[ToString[TeXForm[constant]], ""]]
    |>
  ]
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# ============================================================================
# TIER 4: Interactive Debugging & Tracing
# ============================================================================


@mcp.tool()
async def trace_evaluation(expression: str, max_depth: int = 5) -> str:
    """Trace the step-by-step evaluation of an expression."""
    result = _try_addon_command(
        "trace_evaluation", {"expression": expression, "max_depth": max_depth}
    )

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def time_expression(expression: str) -> str:
    """Time the evaluation of an expression with memory tracking."""
    result = _try_addon_command("time_expression", {"expression": expression})

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def check_syntax(code: str) -> str:
    """Validate Wolfram Language code syntax without executing it."""
    result = _try_addon_command("check_syntax", {"code": code})

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def suggest_similar_functions(query: str) -> str:
    """Find Wolfram functions similar to a query using fuzzy matching."""
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    code = f'''
Module[{{query, matches}},
  query = "{query}";
  matches = Select[
    Names["System`*"],
    StringContainsQ[#, query, IgnoreCase -> True] &
  ];
  matches = Take[matches, UpTo[20]];
  <|
    "success" -> True,
    "query" -> query,
    "matches" -> Map[
      <|
        "name" -> #,
        "usage" -> StringTake[
          ToString[ToExpression[# <> "::usage"] /. _MessageName -> ""],
          UpTo[100]
        ]
      |> &,
      matches
    ]
  |>
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# ============================================================================
# TIER 5: Data I/O
# ============================================================================


@mcp.tool()
async def import_data(
    path: str,
    format: Optional[str] = None,
) -> str:
    """Import data from a file or URL into Mathematica."""
    expanded = _expand_path(path) if not path.startswith("http") else path

    result = _try_addon_command(
        "import_data", {"path": expanded, "format": format or "Automatic"}
    )

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def export_data(
    expression: str,
    path: str,
    format: Optional[str] = None,
) -> str:
    """Export data or graphics to a file."""
    expanded = _expand_path(path)

    result = _try_addon_command(
        "export_data",
        {"expression": expression, "path": expanded, "format": format or "Automatic"},
    )

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def list_supported_formats() -> str:
    """List all supported import/export formats."""
    result = _try_addon_command("list_import_formats", {})

    if result.get("error"):
        # Fallback to wolframscript
        import subprocess
        import shutil

        wolframscript = shutil.which("wolframscript")
        if not wolframscript:
            return json.dumps(
                {"success": False, "error": "wolframscript not found"}, indent=2
            )

        code = """
<|
  "success" -> True,
  "import_formats" -> $ImportFormats,
  "export_formats" -> $ExportFormats,
  "import_count" -> Length[$ImportFormats],
  "export_count" -> Length[$ExportFormats]
|>
"""
        try:
            result = subprocess.run(
                [wolframscript, "-code", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip()
            parsed = _parse_wolfram_association(output)
            return json.dumps(parsed, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    return json.dumps(result, indent=2)


# ============================================================================
# TIER 6: Visualization
# ============================================================================


@mcp.tool()
async def inspect_graphics(expression: str) -> str:
    """Analyze the structure of a graphics object."""
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    code = f'''
Module[{{g, result}},
  g = Quiet[Check[ToExpression["{expression}"], $Failed]];
  If[g === $Failed || !MatchQ[Head[g], Graphics|Graphics3D|Graph|GeoGraphics],
    <|"success" -> False, "error" -> "Not a graphics object"|>,
    <|
      "success" -> True,
      "head" -> ToString[Head[g]],
      "primitives" -> Union[Cases[g, h_Symbol /; Context[h] === "System`" :> ToString[h], Infinity]],
      "options" -> ToString[Options[g], InputForm],
      "plot_range" -> ToString[Quiet[PlotRange /. AbsoluteOptions[g]], InputForm],
      "image_size" -> ToString[Quiet[ImageSize /. AbsoluteOptions[g]], InputForm]
    |>
  ]
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def export_graphics(
    expression: str,
    path: str,
    format: Literal["PNG", "PDF", "SVG", "EPS", "JPEG"] = "PNG",
    size: int = 600,
) -> str:
    """Export a graphics expression to an image file."""
    expanded = _expand_path(path)

    result = _try_addon_command(
        "export_graphics",
        {"expression": expression, "path": expanded, "format": format, "size": size},
    )

    if result.get("error"):
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    return json.dumps(result, indent=2)


@mcp.tool()
async def compare_plots(
    expressions: List[str], labels: Optional[List[str]] = None
) -> str:
    """Generate a side-by-side comparison of multiple plots."""
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    plots_list = "{" + ", ".join(expressions) + "}"
    labels_code = (
        "None" if not labels else "{" + ", ".join(f'"{l}"' for l in labels) + "}"
    )

    code = f"""
Module[{{plots, labels, grid}},
  plots = {plots_list};
  labels = {labels_code};
  grid = If[labels === None,
    GraphicsRow[plots],
    GraphicsRow[MapThread[Labeled[#1, #2, Top] &, {{plots, labels}}]]
  ];
  <|
    "success" -> True,
    "combined_expression" -> ToString[grid, InputForm],
    "plot_count" -> Length[plots]
  |>
]
"""

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def create_animation(
    expression: str,
    parameter: str,
    range_spec: str,
    frames: int = 20,
) -> str:
    """Create an animation by varying a parameter."""
    import subprocess
    import shutil

    wolframscript = shutil.which("wolframscript")
    if not wolframscript:
        return json.dumps(
            {"success": False, "error": "wolframscript not found"}, indent=2
        )

    code = f'''
Module[{{expr, param, range, anim}},
  expr = Hold[{expression}];
  range = {{{parameter}, {range_spec}}};
  anim = Table[
    ReleaseHold[expr /. {parameter} -> val],
    {{val, range[[2]], range[[3]], (range[[3]] - range[[2]])/{frames}}}
  ];
  <|
    "success" -> True,
    "frame_count" -> Length[anim],
    "parameter" -> "{parameter}",
    "animation_expression" -> ToString[ListAnimate[anim], InputForm]
  |>
]
'''

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout.strip()
        parsed = _parse_wolfram_association(output)
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


# ============================================================================
# Feature Flags and Telemetry
# ============================================================================


@mcp.tool()
async def get_feature_status() -> str:
    """Get the status of all feature flags."""
    return json.dumps(
        {
            "success": True,
            "features": FEATURES.to_dict(),
        },
        indent=2,
    )


@mcp.tool()
async def get_telemetry_stats() -> str:
    """Get usage statistics for tools (if telemetry is enabled)."""
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
    """Reset all telemetry statistics to zero."""
    if not FEATURES.telemetry:
        return json.dumps(
            {"success": False, "error": "Telemetry is disabled"}, indent=2
        )

    reset_stats()
    return json.dumps(
        {"success": True, "message": "Telemetry statistics reset"}, indent=2
    )


@mcp.prompt()
def mathematica_expert(user_request: str = "") -> str:
    """
    Expert guidance for using Mathematica tools effectively.
    Use this prompt to determine the best strategy for solving mathematical problems.
    """
    return f"""You are a Mathematica expert with access to a powerful Wolfram Engine integration.

USER REQUEST: {user_request}

### CORE CAPABILITIES
1. **Symbolic Math**: Calculus, Algebra, Discrete Math (Solve, Integrate, D, Sum)
2. **Visualizations**: 2D/3D Plots, Graphs, Animations (Plot, Graphics3D)
3. **Data Analysis**: Statistics, Datasets, Import/Export
4. **Knowledge**: Wolfram Alpha integration (natural language queries)
5. **Notebook Automation**: Create, edit, and run .nb notebooks

### EXECUTION STRATEGY (CRITICAL)

**1. CHOOSE THE RIGHT TARGET (`output_target`):**

*   **USE `output_target="cli"` (Command Line) WHEN:**
    *   Goal is a *result* (number, formula, text).
    *   "Solve x^2=4", "Integrate sin(x)", "Factor 12345".
    *   You need the output to use in subsequent reasoning.
    *   No GUI is available or needed.

*   **USE `output_target="notebook"` (Interactive Notebook) WHEN:**
    *   Goal is a *visual* (Plot, Graphics, Image).
    *   "Plot sin(x)", "Show me a torus", "Visualize the dataset".
    *   User wants to save the work in a .nb file.
    *   **PREREQUISITE**: Mathematica Desktop MUST be running with `StartMCPServer[]` executed.
    *   *If `output_target="notebook"` fails, tools will auto-fallback to CLI.*

**2. HANDLING LONG COMPUTATIONS:**
*   Use `submit_computation` for tasks taking > 60s (large integrals, optimization).
*   Use `background_task` agent for parallel exploration.

**3. NOTEBOOK OPERATIONS:**
*   `create_notebook`: Start fresh.
*   `write_cell`: Add code/text.
*   `evaluate_cell`/`evaluate_selection`: Run code.
*   `screenshot_notebook`: Verify visuals.

### BEST PRACTICES
*   **Format**: Use `InputForm` (standard text) for input. output can be `latex` for math heavy display.
*   **Verification**: For complex derivations, use `verify_derivation`.
*   **Search**: Use `resolve_function` or `wolfram_alpha` if unsure about syntax.

### EXAMPLE WORKFLOWS

**Visual Task ("Plot a 3D surface")**:
1. `create_notebook(title="3D Plot")`
2. `write_cell(content="Plot3D[Sin[x*y], {{x,0,3}}, {{y,0,3}}]", style="Input")`
3. `evaluate_selection()`
4. `screenshot_notebook()` (to confirm)

**Calculation Task ("Derive the volume of a sphere")**:
1. `execute_code("Integrate[1, {{x,y,z}} \\[Element] Ball[]]", output_target="cli")`
"""


def main():
    mcp.run()


if __name__ == "__main__":
    main()
