import asyncio
import json
import logging
import os
import re
import difflib
from typing import Callable, Literal, Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP, Image

from .connection import get_mathematica_connection, close_connection
from .session import (
    execute_in_kernel,
    get_kernel_session,
    close_kernel_session,
    _is_graphics_output,
)
from .config import FEATURES
from .guidance import (
    build_mathematica_expert_prompt,
    create_notebook_doc,
    evaluate_cell_doc,
    evaluate_selection_doc,
    execute_code_doc,
    legacy_notebook_doc,
    read_notebook_doc,
    write_cell_doc,
)
from .telemetry import get_usage_stats, reset_stats
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
_CORE_TOOL_REGISTRY: List[tuple[str, Callable[..., Any]]] = []


def _tool(group: str):
    """Tag a core tool with its exposure group."""

    def decorator(func):
        _CORE_TOOL_REGISTRY.append((group, func))
        return func

    return decorator


def _register_core_tools() -> None:
    for group, func in _CORE_TOOL_REGISTRY:
        if FEATURES.tool_group_enabled(group):
            mcp.tool()(func)


def _json_response(payload: Any) -> str:
    return json.dumps(payload, indent=2)


async def _run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def _addon_result(command: str, params: Optional[dict] = None) -> dict:
    return await _run_blocking(_try_addon_command, command, params)


async def _addon_json(command: str, params: Optional[dict] = None) -> str:
    return _json_response(await _addon_result(command, params))


async def _image_from_result(result: dict) -> Image:
    image_path = result["path"]

    def _read_and_remove() -> bytes:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        os.remove(image_path)
        return image_bytes

    image_bytes = await _run_blocking(_read_and_remove)
    return Image(data=image_bytes, format="png")


def _attach_image_if_valid(result: dict) -> None:
    """Attach rendered_image to result if image_path exists and is non-empty.

    Validates the file on disk before trusting the path.  Strips is_graphics
    and image_path from the result if the file is missing or empty so the
    caller does not hand out a broken path.
    """
    if not result.get("is_graphics") or not result.get("image_path"):
        return
    image_path = result["image_path"]
    if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
        result["rendered_image"] = image_path
        result["tip"] = "Use Read tool to view image."
    else:
        # Rasterization claimed success but no valid file — clean up
        result.pop("image_path", None)
        result.pop("is_graphics", None)
        result.pop("rendered_image", None)
        result.pop("tip", None)
        # Restore output to the textual form so clients don't see a
        # broken "[Graphics rendered to image: ...]" placeholder.
        if result.get("output_inputform"):
            result["output"] = result["output_inputform"]


def _lookup_symbols_in_kernel(query: str) -> Dict[str, Any]:
    """Search for Wolfram symbols matching query via wolframscript."""
    import subprocess
    import shutil

    from .lazy_wolfram_tools import _find_wolframscript
    wolframscript = _find_wolframscript()
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


@_tool("core")
async def get_mathematica_status() -> str:
    """Get connection status and system info."""
    try:
        result = await _addon_result("get_status")
        if result.get("success") is False and result.get("error"):
            raise RuntimeError(result["error"])
        result["connection_mode"] = "addon"
        return _json_response(result)
    except Exception as e:
        try:
            session = await _run_blocking(get_kernel_session)
            if session is None:
                raise RuntimeError("No kernel session available")
            from wolframclient.language import wlexpr

            version = session.evaluate(wlexpr("$VersionNumber"))
            return _json_response(
                {
                    "connection_mode": "kernel_only",
                    "kernel_version": float(version),
                    "note": "Addon not running - notebook control unavailable. Execute StartMCPServer[] in Mathematica.",
                    "error": str(e),
                }
            )
        except Exception as e2:
            return _json_response(
                {
                    "connection_mode": "disconnected",
                    "error": f"No connection available: {e2}",
                }
            )


@_tool("notebook_primary")
async def get_notebooks() -> str:
    """List all open Mathematica notebooks. Returns ID, filename, title."""
    return await _addon_json("get_notebooks")


@_tool("notebook_primary")
async def get_notebook_info(
    notebook: Optional[str] = None, session_id: Optional[str] = None
) -> str:
    """Get details about a notebook (filename, directory, cell count)."""
    result = await _addon_result(
        "get_notebook_info", {"notebook": notebook, "session_id": session_id}
    )
    return _json_response(result)


@_tool("notebook_advanced")
async def create_notebook(title: str = "Untitled", session_id: Optional[str] = None) -> str:
    """Create a new empty notebook. Returns notebook ID.
    
    NOTE: For executing code in a notebook, prefer execute_code(code, output_target="notebook")
    which handles notebook creation, cell writing, and evaluation atomically.
    """
    result = await _addon_result(
        "create_notebook", {"title": title, "session_id": session_id}
    )
    return _json_response(result)


@_tool("notebook_primary")
async def save_notebook(
    notebook: Optional[str] = None,
    path: Optional[str] = None,
    format: Literal["Notebook", "PDF", "HTML", "TeX"] = "Notebook",
    session_id: Optional[str] = None,
) -> str:
    """Save a notebook to disk."""
    result = await _addon_result(
        "save_notebook",
        {
            "notebook": notebook,
            "path": path,
            "format": format,
            "session_id": session_id,
        },
    )
    return _json_response(result)


@_tool("notebook_primary")
async def close_notebook(
    notebook: Optional[str] = None, session_id: Optional[str] = None
) -> str:
    """Close a notebook."""
    result = await _addon_result(
        "close_notebook", {"notebook": notebook, "session_id": session_id}
    )
    return _json_response(result)


@_tool("notebook_primary")
async def get_cells(
    notebook: Optional[str] = None,
    style: Optional[str] = None,
    session_id: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
    include_content: bool = True,
) -> str:
    """Get list of cells in a notebook."""
    result = await _addon_result(
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
    return _json_response(result)


@_tool("notebook_primary")
async def get_cell_content(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Get the full content of a specific cell."""
    result = await _addon_result(
        "get_cell_content",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return _json_response(result)


@_tool("notebook_advanced")
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
    result = await _addon_result(
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
    return _json_response(result)


@_tool("notebook_advanced")
async def delete_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Delete a cell from a notebook."""
    result = await _addon_result(
        "delete_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return _json_response(result)


@_tool("notebook_advanced")
async def evaluate_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
    max_wait: int = 10,
    sync: Literal["none", "refresh", "strict"] = "none",
    sync_wait: float = 2,
) -> str:
    """Evaluate a specific cell."""
    result = await _addon_result(
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
    return _json_response(result)


@_tool("core")
async def execute_code(
    code: str,
    format: Literal["text", "latex", "mathematica"] = "text",
    output_target: Optional[Literal["cli", "notebook"]] = None,
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
    """Execute Wolfram Language code."""
    if output_target is None:
        output_target = FEATURES.default_output_target

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
            result = await _addon_result("execute_code_notebook", params)

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

                return _json_response(response)
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
                    "render_graphics": render_graphics,
                    "session_id": session_id,
                    "isolate_context": isolate_context,
                    "timeout": timeout,
                }
                if deterministic_seed is not None:
                    params["deterministic_seed"] = deterministic_seed
                result = await _addon_result("execute_code", params)
                if isinstance(result, dict):
                    result["note"] = "Executed via CLI (notebook error)."
                    _attach_image_if_valid(result)
                    return _json_response(result)
                return f"{result}\n(Note: Executed via CLI due to notebook error)"
            except Exception:
                result = await _run_blocking(
                    execute_in_kernel,
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
                _attach_image_if_valid(result)
                return _json_response(result)

    params = {
        "code": code,
        "format": format,
        "render_graphics": render_graphics,
        "session_id": session_id,
        "isolate_context": isolate_context,
        "timeout": timeout,
    }
    if deterministic_seed is not None:
        params["deterministic_seed"] = deterministic_seed
    result = await _addon_result("execute_code", params)
    # Check if addon command succeeded, otherwise fall back to kernel
    if result.get("success") is False or "error" in result:
        result = await _run_blocking(
            execute_in_kernel,
            code,
            format,
            render_graphics=render_graphics,
            deterministic_seed=deterministic_seed,
            session_id=session_id,
            isolate_context=isolate_context,
            timeout=timeout,
        )
        result["execution_mode"] = "kernel_fallback"
    _attach_image_if_valid(result)
    return _json_response(result)


@_tool("admin")
async def batch_commands(commands: List[Dict[str, Any]]) -> str:
    """Execute multiple commands in one round-trip."""
    return await _addon_json("batch_commands", {"commands": commands})


@_tool("notebook_advanced")
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
    result = await _addon_result(
        "execute_selection",
        {
            "notebook": notebook,
            "session_id": session_id,
            "max_wait": max_wait,
            "sync": sync,
            "sync_wait": sync_wait,
        },
    )
    return _json_response(result)


@_tool("notebook_primary")
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
    result = await _addon_result(
        "screenshot_notebook",
        {
            "notebook": notebook,
            "max_height": max_height,
            "session_id": session_id,
            "use_rasterize": use_rasterize,
            "wait_ms": wait_ms,
        },
    )

    return await _image_from_result(result)


@_tool("notebook_primary")
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
    result = await _addon_result(
        "screenshot_cell",
        {
            "cell_id": cell_id,
            "notebook": notebook,
            "session_id": session_id,
            "use_rasterize": use_rasterize,
        },
    )

    return await _image_from_result(result)


@_tool("notebook_primary")
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
    result = await _addon_result(
        "rasterize_expression", {"expression": expression, "image_size": image_size}
    )
    return await _image_from_result(result)


@_tool("notebook_advanced")
async def select_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Select a cell in the notebook (moves cursor to it)."""
    result = await _addon_result(
        "select_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return _json_response(result)


@_tool("notebook_advanced")
async def scroll_to_cell(
    cell_id: str,
    notebook: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Scroll the notebook view to make a cell visible."""
    result = await _addon_result(
        "scroll_to_cell",
        {"cell_id": cell_id, "notebook": notebook, "session_id": session_id},
    )
    return _json_response(result)


@_tool("notebook_primary")
async def export_notebook(
    path: str,
    notebook: Optional[str] = None,
    format: Literal["PDF", "HTML", "TeX", "Markdown"] = "PDF",
    session_id: Optional[str] = None,
) -> str:
    """Export a notebook to PDF, HTML, TeX, or Markdown."""
    result = await _addon_result(
        "export_notebook",
        {"notebook": notebook, "path": path, "format": format, "session_id": session_id},
    )
    return _json_response(result)


@_tool("debug")
async def verify_derivation(
    steps: List[str],
    format: Literal["text", "latex", "mathematica"] = "text",
    timeout: int = 120,
) -> str:
    """Verify a sequence of mathematical expressions steps."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.verify_derivation(
        steps,
        format,
        timeout,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("core")
async def get_kernel_state() -> str:
    """Get current Wolfram kernel session state (memory, uptime, version)."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.get_kernel_state(
        parse_wolfram_association=_parse_wolfram_association
    )


@_tool("kernel_tools")
async def load_package(package_name: str) -> str:
    """Load a Mathematica package (e.g., "Developer`")."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.load_package(
        package_name,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("kernel_tools")
async def list_loaded_packages() -> str:
    """List all currently loaded packages and contexts."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.list_loaded_packages(
        parse_wolfram_association=_parse_wolfram_association
    )


# ============================================================================
# TIER 1: Variable Introspection (Session State)
# ============================================================================


@_tool("session")
async def list_variables(include_system: bool = False) -> str:
    """List all user-defined variables in the current Mathematica kernel session."""
    result = await _addon_result("list_variables", {"include_system": include_system})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


@_tool("session")
async def get_variable(name: str) -> str:
    """Get detailed information about a specific variable."""
    result = await _addon_result("get_variable", {"name": name})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


@_tool("session")
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
    result = await _addon_result("set_variable", {"name": name, "value": value})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


@_tool("session")
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

    result = await _addon_result("clear_variables", params)

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


@_tool("session")
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
    result = await _addon_result("get_expression_info", {"expression": expression})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


# ============================================================================
# TIER 1: Error Recovery
# ============================================================================


@_tool("core")
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
    result = await _addon_result("get_messages", {"count": count})

    if result.get("error"):
        return _json_response({"success": False, "error": result["error"]})

    return _json_response(result)


@_tool("core")
async def restart_kernel() -> str:
    """
    Restart the Mathematica kernel, clearing all state.

    This is the nuclear option - clears all variables, definitions, and state.
    Use when the kernel is in a bad state or you need a fresh start.

    Returns:
        Confirmation of kernel restart
    """
    await _run_blocking(close_kernel_session)
    # Force reconnection
    result = await _addon_result("ping")

    return _json_response(
        {
            "success": True,
            "message": "Kernel session cleared. Fresh session will be created on next execution.",
            "ping_result": result,
        }
    )


# ============================================================================
# TIER 2: File Handling (.nb, .wl, .wlnb)
# ============================================================================


def _expand_path(path: str) -> str:
    """Expand ~ and make path absolute."""
    expanded = os.path.expanduser(path)
    return os.path.abspath(expanded)


def _load_cached_notebook(path: str, truncation_threshold: int = 25000):
    from .notebook_parser import parse_notebook_cached

    return parse_notebook_cached(path, truncation_threshold=truncation_threshold)


def _register_optional_tools() -> None:
    if FEATURES.symbol_lookup:
        from .optional_symbol_tools import register_symbol_lookup_tools

        register_symbol_lookup_tools(
            mcp,
            lookup_symbols_in_kernel=_lookup_symbols_in_kernel,
            extract_short_description=_extract_short_description,
            extract_example_signature=_extract_example_signature,
            rank_candidates=_rank_candidates,
            parse_wolfram_association=_parse_wolfram_association,
            execute_code=execute_code,
        )

    if FEATURES.math_aliases:
        from .optional_math_aliases import register_math_alias_tools

        register_math_alias_tools(mcp, execute_code)

    if FEATURES.function_repository:
        from .optional_repository_tools import register_function_repository_tools

        register_function_repository_tools(
            mcp, parse_wolfram_association=_parse_wolfram_association
        )

    if FEATURES.data_repository:
        from .optional_repository_tools import register_data_repository_tools

        register_data_repository_tools(
            mcp, parse_wolfram_association=_parse_wolfram_association
        )

    if FEATURES.async_computation:
        from .optional_async_jobs import register_async_computation_tools

        register_async_computation_tools(mcp)

    if FEATURES.cache_tools:
        from .optional_cache_tools import register_cache_tools

        register_cache_tools(
            mcp,
            cache_expression_fn=_cache_expr,
            get_cached_expression_fn=_get_cached,
            list_cached_expressions_fn=list_cached_expressions,
            clear_cache_fn=clear_cache,
            execute_code=execute_code,
        )

    if FEATURES.telemetry:
        from .optional_telemetry_tools import register_telemetry_tools

        register_telemetry_tools(
            mcp,
            get_usage_stats=get_usage_stats,
            reset_stats=reset_stats,
        )


def _apply_guidance_docs() -> None:
    execute_code.__doc__ = execute_code_doc(FEATURES.default_output_target)
    create_notebook.__doc__ = create_notebook_doc()
    write_cell.__doc__ = write_cell_doc()
    evaluate_cell.__doc__ = evaluate_cell_doc()
    evaluate_selection.__doc__ = evaluate_selection_doc()
    read_notebook.__doc__ = read_notebook_doc()
    read_notebook_content.__doc__ = legacy_notebook_doc(
        "Read notebook content as structured text."
    )
    convert_notebook.__doc__ = legacy_notebook_doc(
        "Convert a notebook into markdown, LaTeX, plain text, or Wolfram code."
    )
    get_notebook_outline.__doc__ = legacy_notebook_doc(
        "Get the notebook's section outline."
    )
    parse_notebook_python.__doc__ = legacy_notebook_doc(
        "Parse a notebook with the Python-native parser."
    )
    get_notebook_cell.__doc__ = legacy_notebook_doc(
        "Read a single notebook cell by index."
    )


@_tool("notebook_primary")
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
    result = await _addon_result(
        "open_notebook_file", {"path": expanded, "session_id": session_id}
    )

    if result.get("error"):
        return _json_response(
            {"success": False, "error": result["error"], "path": expanded}
        )

    return _json_response(result)


@_tool("file_legacy")
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
    result = await _addon_result("run_script", {"path": expanded})

    if result.get("error"):
        return _json_response(
            {"success": False, "error": result["error"], "path": expanded}
        )

    return _json_response(result)


@_tool("file_legacy")
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
    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    try:
        from .notebook_parser import CellStyle

        notebook = await _run_blocking(
            _load_cached_notebook, expanded, truncation_threshold=25000
        )
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

        return _json_response(
            {
                "success": True,
                "path": expanded,
                "cell_count": len(cells),
                "cells": cells[:50],
            }
        )
    except Exception as e:
        return _json_response({"success": False, "error": str(e)})


@_tool("file_legacy")
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
    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    try:
        from .notebook_parser import NotebookParser

        notebook = await _run_blocking(
            _load_cached_notebook, expanded, truncation_threshold=25000
        )
        parser = NotebookParser(truncation_threshold=25000)

        if output_format == "markdown":
            content = parser.to_markdown(notebook)
        elif output_format == "wolfram":
            content = parser.to_wolfram_code(notebook)
        elif output_format == "plain":
            content = parser.to_plain_text(notebook)
        else:
            import shutil
            import subprocess

            from .lazy_wolfram_tools import _find_wolframscript
            wolframscript = _find_wolframscript()
            if not wolframscript:
                return _json_response({"success": False, "error": "wolframscript not found"})

            code = f'''
Module[{{nb, content}},
  nb = Import["{expanded}"];
  content = ExportString[nb, "TeX"];
  <|"success" -> True, "format" -> "{output_format}", "content" -> content|>
]
'''
            result = await _run_blocking(
                subprocess.run,
                [wolframscript, "-code", code],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout.strip()
            parsed = _parse_wolfram_association(output)
            return _json_response(parsed)

        return _json_response(
            {"success": True, "format": output_format, "content": content}
        )
    except Exception as e:
        return _json_response({"success": False, "error": str(e)})


@_tool("file_legacy")
async def get_notebook_outline(path: str) -> str:
    """
    Get the structural outline of a notebook (sections, subsections, titles).

    Like a table of contents - shows the organization without full content.

    Args:
        path: Path to the .nb file

    Returns:
        Hierarchical outline of notebook sections
    """
    expanded = _expand_path(path)

    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    try:
        notebook = await _run_blocking(
            _load_cached_notebook, expanded, truncation_threshold=25000
        )
        sections = notebook.get_outline()
        return _json_response(
            {
                "success": True,
                "path": expanded,
                "sections": sections,
                "count": len(sections),
            }
        )
    except Exception as e:
        return _json_response({"success": False, "error": str(e)})


@_tool("file_legacy")
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
        effective_threshold = truncation_threshold if truncation_threshold > 0 else 10**9
        parser = NotebookParser(truncation_threshold=effective_threshold)
        notebook = await _run_blocking(
            _load_cached_notebook,
            expanded,
            truncation_threshold=effective_threshold,
        )

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


@_tool("file_legacy")
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
        notebook = await _run_blocking(
            _load_cached_notebook, expanded, truncation_threshold=threshold
        )

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
# TIER 2b: Consolidated Notebook Reading (backend-aware)
# ============================================================================


@_tool("notebook_primary")
async def read_notebook(
    path: str,
    output_format: Literal[
        "markdown", "wolfram", "outline", "json", "plain"
    ] = "markdown",
    cell_types: Optional[List[str]] = None,
    include_outputs: bool = True,
    backend: Optional[str] = None,
    view: str = "semantic",
    include_alternates: bool = False,
    truncation_threshold: int = 25000,
) -> str:
    """
    Read a Mathematica notebook with capability-based backend dispatch.

    Consolidates notebook reading into a single tool. Uses the best available
    backend: kernel (accurate, via NotebookImport) or Python parser (offline).

    Args:
        path: Path to the .nb file
        output_format:
            - "markdown": Readable Markdown with code blocks (default)
            - "wolfram": Pure executable Wolfram Language code only
            - "outline": Hierarchical section outline
            - "json": Structured JSON with cell data and metadata
            - "plain": Plain text
        cell_types: Optional filter — list of styles like ["Input", "Text", "Section"].
                    If omitted, all cell types are included.
        include_outputs: If True (default), include Output cells.
                         If False, filters out Output/Message/Print cells.
        backend: Force a specific backend: "python_syntax" or "kernel_semantic".
                 If omitted, auto-selects based on capability and availability.
        view: Primary view mode: "semantic" (default), "display", or "raw"
        include_alternates: If True, include alternate views per cell (JSON only)
        truncation_threshold: Max chars per cell before truncation (0 = no limit)

    Returns:
        Notebook content in the requested format
    """
    from .notebook_backend import extract_notebook, CellView

    expanded = _expand_path(path)
    if not os.path.exists(expanded):
        return json.dumps(
            {"success": False, "error": f"File not found: {expanded}"}, indent=2
        )

    # Build cell type filter
    effective_types = list(cell_types) if cell_types else None
    if not include_outputs:
        # Remove output-like styles from whatever type list we have
        output_styles = {"Output", "Message", "Print"}
        if effective_types is not None:
            effective_types = [t for t in effective_types if t not in output_styles]
        else:
            # No explicit types: include everything except output styles
            effective_types = [
                "Input", "Code", "Text", "Title", "Chapter",
                "Section", "Subsection", "Subsubsection",
                "Item", "ItemNumbered",
            ]

    # Map format to capability hint for dispatch
    capability_map = {
        "wolfram": "code",
        "outline": "outline",
        "plain": "text",
        "markdown": "full",
        "json": "full",
    }
    capability = capability_map.get(output_format, "full")

    view_enum = {
        "semantic": CellView.SEMANTIC,
        "display": CellView.DISPLAY,
        "raw": CellView.RAW,
    }.get(view, CellView.SEMANTIC)

    try:
        result = await extract_notebook(
            expanded,
            capability=capability,
            cell_types=effective_types,
            view=view_enum,
            include_alternates=include_alternates,
            truncation_threshold=truncation_threshold,
            force_backend=backend,
        )

        if result.error:
            return _json_response({"success": False, "error": result.error})

        if output_format == "markdown":
            return _json_response({
                "success": True,
                "format": "markdown",
                "backend": result.backend,
                "path": expanded,
                "cell_count": result.cell_count,
                "code_cells": result.code_cell_count,
                "content": result.to_markdown(),
            })
        elif output_format == "wolfram":
            return _json_response({
                "success": True,
                "format": "wolfram",
                "backend": result.backend,
                "path": expanded,
                "code_cells": result.code_cell_count,
                "content": result.to_wolfram_code(),
            })
        elif output_format == "outline":
            outline = result.to_outline()
            return _json_response({
                "success": True,
                "format": "outline",
                "backend": result.backend,
                "path": expanded,
                "section_count": len(outline),
                "sections": outline,
            })
        elif output_format == "plain":
            return _json_response({
                "success": True,
                "format": "plain",
                "backend": result.backend,
                "path": expanded,
                "cell_count": result.cell_count,
                "content": result.to_plain_text(),
            })
        elif output_format == "json":
            return _json_response(result.to_dict(include_alternates))
        else:
            return _json_response({"success": False, "error": f"Unknown format: {output_format}"})

    except Exception as e:
        return _json_response({"success": False, "error": str(e)})


# ============================================================================
# TIER 3: Wolfram Alpha & Natural Language
# ============================================================================


@_tool("knowledge")
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
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.wolfram_alpha(
        query,
        return_type,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("knowledge")
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
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.interpret_natural_language(text)


@_tool("knowledge")
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
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.entity_lookup(
        entity_type,
        name,
        properties,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("knowledge")
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
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.convert_units(
        quantity,
        target_unit,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("knowledge")
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
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.get_constant(
        name, parse_wolfram_association=_parse_wolfram_association
    )


# ============================================================================
# TIER 4: Interactive Debugging & Tracing
# ============================================================================


@_tool("debug")
async def trace_evaluation(expression: str, max_depth: int = 5) -> str:
    """Trace the step-by-step evaluation of an expression."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.trace_evaluation(
        expression,
        max_depth,
        addon_result=_addon_result,
    )


@_tool("debug")
async def time_expression(expression: str) -> str:
    """Time the evaluation of an expression with memory tracking."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.time_expression(
        expression, addon_result=_addon_result
    )


@_tool("core")
async def check_syntax(code: str) -> str:
    """Validate Wolfram Language code syntax without executing it."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.check_syntax(
        code, addon_result=_addon_result
    )


# ============================================================================
# TIER 5: Data I/O
# ============================================================================


@_tool("data")
async def import_data(
    path: str,
    format: Optional[str] = None,
) -> str:
    """Import data from a file or URL into Mathematica."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.import_data(
        path,
        format,
        addon_result=_addon_result,
        expand_path=_expand_path,
    )


@_tool("data")
async def export_data(
    expression: str,
    path: str,
    format: Optional[str] = None,
) -> str:
    """Export data or graphics to a file."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.export_data(
        expression,
        path,
        format,
        addon_result=_addon_result,
        expand_path=_expand_path,
    )


@_tool("data")
async def list_supported_formats() -> str:
    """List all supported import/export formats."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.list_supported_formats(
        addon_result=_addon_result,
        parse_wolfram_association=_parse_wolfram_association,
    )


# ============================================================================
# TIER 6: Visualization
# ============================================================================


@_tool("graphics")
async def inspect_graphics(expression: str) -> str:
    """Analyze the structure of a graphics object."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.inspect_graphics(
        expression, parse_wolfram_association=_parse_wolfram_association
    )


@_tool("graphics")
async def export_graphics(
    expression: str,
    path: str,
    format: Literal["PNG", "PDF", "SVG", "EPS", "JPEG"] = "PNG",
    size: int = 600,
) -> str:
    """Export a graphics expression to an image file."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.export_graphics(
        expression,
        path,
        format,
        size,
        addon_result=_addon_result,
        expand_path=_expand_path,
    )


@_tool("graphics")
async def compare_plots(
    expressions: List[str], labels: Optional[List[str]] = None
) -> str:
    """Generate a side-by-side comparison of multiple plots."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.compare_plots(
        expressions,
        labels,
        parse_wolfram_association=_parse_wolfram_association,
    )


@_tool("graphics")
async def create_animation(
    expression: str,
    parameter: str,
    range_spec: str,
    frames: int = 20,
) -> str:
    """Create an animation by varying a parameter."""
    from . import lazy_wolfram_tools as _lazy_wolfram_tools

    return await _lazy_wolfram_tools.create_animation(
        expression,
        parameter,
        range_spec,
        frames,
        parse_wolfram_association=_parse_wolfram_association,
    )


# ============================================================================
# Feature Flags and Telemetry
# ============================================================================


@_tool("core")
async def get_feature_status() -> str:
    """Get the status of all feature flags."""
    return json.dumps(
        {
            "success": True,
            "features": FEATURES.to_dict(),
        },
        indent=2,
    )


_apply_guidance_docs()
_register_core_tools()
_register_optional_tools()


@mcp.prompt()
def mathematica_expert(user_request: str = "") -> str:
    """Expert guidance for using Mathematica tools effectively."""
    return build_mathematica_expert_prompt(user_request, features=FEATURES)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
