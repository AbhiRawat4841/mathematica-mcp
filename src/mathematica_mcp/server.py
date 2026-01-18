import json
import logging
import os
from typing import Literal, Optional

from mcp.server.fastmcp import FastMCP, Image

from .connection import get_mathematica_connection, close_connection
from .session import execute_in_kernel, get_kernel_session, close_kernel_session

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mathematica_mcp")

mcp = FastMCP("mathematica-mcp")


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
    output_target: Literal["cli", "notebook"] = "cli",
) -> str:
    """
    Execute Wolfram Language code and return the result.

    Args:
        code: Wolfram Language code to execute
        format: Output format - text (default), latex (for equations), mathematica (InputForm)
        output_target: Where to display the output - "cli" (return text) or "notebook" (insert into active notebook)

    Examples:
        execute_code("Integrate[x^2, x]")
        execute_code("Plot[Sin[x], {x,0,Pi}]", output_target="notebook")
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


def main():
    mcp.run()


if __name__ == "__main__":
    main()
