"""
Notebook extraction backend abstraction.

Provides capability-based dispatch across three backends:
- PythonSyntaxBackend: offline, fast, approximate (existing parser)
- KernelSemanticBackend: headless kernel via NotebookImport, accurate
- AddonLiveBackend: requires frontend, for interactive operations only

Design principles (from architectural review):
- Dispatch by requested capability, not backend availability
- Internal model preserves enough to derive raw/semantic/display views
- Default response returns exactly one primary view per cell
- Alternates are opt-in, not default
- Don't throw away information too early
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("mathematica_mcp.notebook_backend")


# ---------------------------------------------------------------------------
# Cell model
# ---------------------------------------------------------------------------


class CellView(Enum):
    """View modes for cell content."""

    SEMANTIC = "semantic"  # Wolfram InputForm for code, meaningful text for prose
    DISPLAY = "display"  # Human-readable / formatted text
    RAW = "raw"  # Raw BoxData / source as-is


@dataclass
class NotebookCell:
    """Rich cell representation with optional alternate views.

    Default response returns a single primary ``content`` field.
    Alternate views are stored internally and returned only on request
    (``include_alternates=True``) or when conversion is lossy.
    """

    cell_index: int
    style: str  # "Input", "Output", "Text", "Section", etc.
    content: str  # Primary view (semantic for code, display for prose)

    # Identity & provenance
    cell_id: int | None = None
    group_id: int | None = None
    position_in_group: int = 1
    parent_input_index: int | None = None
    cell_label: str = ""
    tags: list[str] = field(default_factory=list)
    is_generated: bool = False

    # Truncation metadata
    was_truncated: bool = False
    original_length: int = 0
    conversion_lossy: bool = False

    # Internal: alternate views (not serialized by default)
    _raw: str | None = field(default=None, repr=False)
    _semantic: str | None = field(default=None, repr=False)
    _display: str | None = field(default=None, repr=False)

    def get_view(self, view: CellView) -> str:
        """Get a specific view of this cell's content."""
        if view == CellView.RAW and self._raw is not None:
            return self._raw
        if view == CellView.SEMANTIC and self._semantic is not None:
            return self._semantic
        if view == CellView.DISPLAY and self._display is not None:
            return self._display
        return self.content

    def to_dict(self, include_alternates: bool = False) -> dict[str, Any]:
        """Serialize for JSON response. Lean by default."""
        d: dict[str, Any] = {
            "index": self.cell_index,
            "style": self.style,
            "content": self.content,
        }
        if self.cell_label:
            d["cell_label"] = self.cell_label
        if self.cell_id is not None:
            d["cell_id"] = self.cell_id
        if self.group_id is not None:
            d["group_id"] = self.group_id
            d["position_in_group"] = self.position_in_group
        if self.parent_input_index is not None:
            d["parent_input_index"] = self.parent_input_index
        if self.tags:
            d["tags"] = self.tags
        if self.is_generated:
            d["is_generated"] = True
        if self.was_truncated:
            d["was_truncated"] = True
            d["original_length"] = self.original_length
        if self.conversion_lossy:
            d["conversion_lossy"] = True

        if include_alternates:
            alternates: dict[str, str] = {}
            if self._raw is not None:
                alternates["raw"] = self._raw
            if self._semantic is not None:
                alternates["semantic"] = self._semantic
            if self._display is not None:
                alternates["display"] = self._display
            if alternates:
                d["alternates"] = alternates

        return d


# ---------------------------------------------------------------------------
# Notebook result
# ---------------------------------------------------------------------------


@dataclass
class NotebookResult:
    """Result of notebook extraction from any backend."""

    path: str
    cells: list[NotebookCell]
    backend: str  # "python_syntax", "kernel_semantic", "addon_live"
    title: str | None = None
    error: str | None = None

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def code_cell_count(self) -> int:
        return sum(1 for c in self.cells if c.style in ("Input", "Code"))

    def to_dict(self, include_alternates: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "success": self.error is None,
            "path": self.path,
            "backend": self.backend,
            "cell_count": self.cell_count,
            "code_cells": self.code_cell_count,
        }
        if self.title:
            d["title"] = self.title
        if self.error:
            d["error"] = self.error
        d["cells"] = [c.to_dict(include_alternates) for c in self.cells]
        return d

    # ----- Output format helpers -----

    def to_markdown(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(f"# {self.title}\n")
        for cell in self.cells:
            c = cell.content
            if cell.style == "Title":
                parts.append(f"# {c}\n")
            elif cell.style == "Section":
                parts.append(f"## {c}\n")
            elif cell.style == "Subsection":
                parts.append(f"### {c}\n")
            elif cell.style == "Subsubsection":
                parts.append(f"#### {c}\n")
            elif cell.style == "Text":
                parts.append(f"{c}\n")
            elif cell.style in ("Input", "Code"):
                lbl = f" (* {cell.cell_label} *)" if cell.cell_label else ""
                parts.append(f"```wolfram{lbl}\n{c}\n```\n")
            elif cell.style == "Output":
                if c.strip().startswith("|"):
                    parts.append(f"{c}\n")
                elif "\\" in c and not c.startswith("```"):
                    parts.append(f"$$\n{c}\n$$\n")
                else:
                    parts.append(f"```\n(* Output *)\n{c}\n```\n")
            else:
                if c.strip():
                    parts.append(f"{c}\n")
            if cell.was_truncated:
                parts.append(f"> *(Truncated {cell.original_length} chars)*\n")
        return "\n".join(parts)

    def to_wolfram_code(self) -> str:
        lines: list[str] = []
        for cell in self.cells:
            if cell.style in ("Input", "Code"):
                if cell.cell_label:
                    lines.append(f"(* {cell.cell_label} *)")
                if cell.content.strip():
                    lines.append(cell.content)
                if cell.was_truncated:
                    lines.append(
                        f"(* TRUNCATED {cell.original_length} chars; use get_notebook_cell for full content *)"
                    )
                lines.append("")
        return "\n".join(lines).rstrip()

    def to_outline(self) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        for cell in self.cells:
            if cell.style in ("Title", "Chapter", "Section", "Subsection", "Subsubsection"):
                sections.append(
                    {
                        "level": cell.style,
                        "title": cell.content.strip(),
                        "index": cell.cell_index,
                    }
                )
        return sections

    def to_plain_text(self) -> str:
        lines: list[str] = []
        if self.title:
            lines.append(self.title)
            lines.append("")
        for cell in self.cells:
            if cell.content.strip():
                lines.append(cell.content)
                lines.append("")
        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Backend: Python Syntax (wraps existing notebook_parser.py)
# ---------------------------------------------------------------------------


class PythonSyntaxBackend:
    """Offline parser using existing Python BoxData converter.

    Fast and requires no external dependencies, but may be lossy for
    complex box structures.  Serves as the always-available fallback.
    """

    name = "python_syntax"

    def available(self) -> bool:
        return True

    async def extract(
        self,
        path: str,
        *,
        cell_types: list[str] | None = None,
        view: CellView = CellView.SEMANTIC,
        include_alternates: bool = False,
        truncation_threshold: int = 25000,
    ) -> NotebookResult:
        from .notebook_parser import parse_notebook_cached, truncate_large_expression

        effective = truncation_threshold if truncation_threshold > 0 else 10**9
        notebook = await asyncio.to_thread(
            parse_notebook_cached,
            path,
            truncation_threshold=effective,
        )

        # Map CellStyle enum -> string for filtering
        style_str_set = set(cell_types) if cell_types else None

        cells: list[NotebookCell] = []
        # Track groups: consecutive Input then Output belong together
        last_input_idx: int | None = None

        for old_cell in notebook.cells:
            style = old_cell.style.value  # CellStyle enum -> string

            if style_str_set and style not in style_str_set:
                continue

            # Store all available views internally
            semantic_content = old_cell.content
            raw_content = old_cell.raw_content if old_cell.raw_content else None
            display_content = old_cell.content  # Python parser: same as semantic

            # Select primary content based on requested view
            if view == CellView.RAW and raw_content:
                primary = raw_content
            elif view == CellView.DISPLAY:
                primary = display_content
            else:
                primary = semantic_content

            # Apply truncation to the *selected* primary content,
            # not just inherit from the semantic parse.
            was_truncated = False
            original_length = len(primary)
            if truncation_threshold > 0 and len(primary) > truncation_threshold:
                primary, was_truncated, original_length = truncate_large_expression(primary, truncation_threshold)
            elif old_cell.was_truncated and primary is semantic_content:
                # Semantic was already truncated by the parser
                was_truncated = old_cell.was_truncated
                original_length = old_cell.original_length

            nc = NotebookCell(
                cell_index=old_cell.cell_index,
                style=style,
                content=primary,
                cell_label=old_cell.cell_label,
                was_truncated=was_truncated,
                original_length=original_length,
            )

            # Store alternate views for on-demand access
            if raw_content:
                nc._raw = raw_content
            nc._semantic = semantic_content
            nc._display = display_content

            # Track input→output provenance
            if style in ("Input", "Code"):
                last_input_idx = nc.cell_index
            elif style == "Output" and last_input_idx is not None:
                nc.parent_input_index = last_input_idx

            cells.append(nc)

        return NotebookResult(
            path=path,
            cells=cells,
            backend=self.name,
            title=notebook.title or None,
        )


# ---------------------------------------------------------------------------
# Backend: Kernel Semantic (uses session.py + NotebookImport)
# ---------------------------------------------------------------------------


class KernelSemanticBackend:
    """Headless kernel extraction via NotebookImport.

    Uses the existing session.py infrastructure (WolframLanguageSession
    with wolframscript fallback).  Primary backend for accurate code
    extraction.
    """

    name = "kernel_semantic"

    def __init__(self) -> None:
        self._wl_helper_loaded = False

    def available(self) -> bool:
        """Check if wolframscript or wolframclient is available."""
        from .lazy_wolfram_tools import _find_wolframscript

        if _find_wolframscript():
            return True
        try:
            from wolframclient.evaluation import WolframLanguageSession  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_helper_wl_path(self) -> str:
        """Path to the shipped .wl helper file."""
        return str(Path(__file__).parent / "helpers" / "notebook_converter.wl")

    async def extract(
        self,
        path: str,
        *,
        cell_types: list[str] | None = None,
        view: CellView = CellView.SEMANTIC,
        include_alternates: bool = False,
        truncation_threshold: int = 25000,
    ) -> NotebookResult:
        from .disk_cache import get_cached, put_cached
        from .session import execute_in_kernel

        abs_path = str(Path(path).resolve())

        # Check disk cache first (file-aware: keyed on mtime + size)
        cache_opts = dict(
            truncation_threshold=truncation_threshold,
            view=view.value,
        )
        cached = await asyncio.to_thread(
            get_cached,
            abs_path,
            self.name,
            **cache_opts,
        )
        if cached is not None:
            return self._build_result_from_data(
                cached,
                abs_path,
                cell_types=cell_types,
                view=view,
                truncation_threshold=truncation_threshold,
            )

        # Escape backslashes for Wolfram string literal
        wl_path = abs_path.replace("\\", "\\\\")
        helper_path = self._get_helper_wl_path().replace("\\", "\\\\")

        # Include file mtime in the code string so execute_in_kernel's
        # generic cache is invalidated when the notebook file changes.
        try:
            stat = os.stat(abs_path)
            mtime_ns = stat.st_mtime_ns
        except OSError:
            mtime_ns = 0

        wl_code = f"""
Module[{{result}},
  (* mtime_ns={mtime_ns} — cache-busting sentinel *)
  If[!TrueQ[$MCPNotebookConverterLoaded],
    Get["{helper_path}"];
    $MCPNotebookConverterLoaded = True;
  ];
  MCPNotebookConverter`MCPExtractNotebook["{wl_path}"]
]
"""
        result = await asyncio.to_thread(
            execute_in_kernel,
            wl_code,
            output_format="text",
            render_graphics=False,
            timeout=30,
        )

        if not result.get("success"):
            raise RuntimeError(f"Kernel extraction failed: {result.get('output', 'unknown error')}")

        output = result.get("output", "")

        # Store raw kernel output in disk cache for next time
        try:
            raw_data = json.loads(output)
            if not raw_data.get("error"):
                await asyncio.to_thread(
                    put_cached,
                    abs_path,
                    self.name,
                    raw_data,
                    **cache_opts,
                )
        except (json.JSONDecodeError, OSError):
            pass  # Don't fail extraction if caching fails

        return self._parse_kernel_output(
            output,
            abs_path,
            cell_types=cell_types,
            view=view,
            truncation_threshold=truncation_threshold,
        )

    def _parse_kernel_output(
        self,
        output: str,
        path: str,
        *,
        cell_types: list[str] | None = None,
        view: CellView = CellView.SEMANTIC,
        truncation_threshold: int = 25000,
    ) -> NotebookResult:
        """Parse JSON string output from the WL helper."""
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse kernel output as JSON: {output[:200]}") from e
        return self._build_result_from_data(
            data,
            path,
            cell_types=cell_types,
            view=view,
            truncation_threshold=truncation_threshold,
        )

    def _build_result_from_data(
        self,
        data: dict[str, Any],
        path: str,
        *,
        cell_types: list[str] | None = None,
        view: CellView = CellView.SEMANTIC,
        truncation_threshold: int = 25000,
    ) -> NotebookResult:
        """Build NotebookResult from parsed data dict."""
        from .notebook_parser import truncate_large_expression

        if data.get("error"):
            return NotebookResult(
                path=path,
                cells=[],
                backend=self.name,
                error=data["error"],
            )

        style_filter = set(cell_types) if cell_types else None
        cells: list[NotebookCell] = []
        last_input_idx: int | None = None

        for i, cell_data in enumerate(data.get("cells", [])):
            style = cell_data.get("style", "Unknown")
            if style_filter and style not in style_filter:
                continue

            semantic_content = cell_data.get("content", "")
            was_truncated = False
            original_length = len(semantic_content)

            if truncation_threshold > 0 and len(semantic_content) > truncation_threshold:
                semantic_content, was_truncated, original_length = truncate_large_expression(
                    semantic_content, truncation_threshold
                )

            # Kernel backend only has semantic form — no raw boxes.
            # If raw was requested, mark every cell as lossy.
            is_lossy = cell_data.get("conversion_lossy", False)
            if view == CellView.RAW:
                is_lossy = True

            nc = NotebookCell(
                cell_index=i,
                style=style,
                content=semantic_content,
                cell_id=cell_data.get("cell_id"),
                group_id=cell_data.get("group_id"),
                position_in_group=cell_data.get("position_in_group", 1),
                cell_label=cell_data.get("cell_label", ""),
                tags=cell_data.get("tags", []),
                is_generated=cell_data.get("is_generated", False),
                was_truncated=was_truncated,
                original_length=original_length,
                conversion_lossy=is_lossy,
            )

            # Store semantic as alternate
            nc._semantic = semantic_content

            # Track input→output provenance
            if style in ("Input", "Code"):
                last_input_idx = nc.cell_index
            elif style == "Output" and last_input_idx is not None:
                nc.parent_input_index = last_input_idx

            cells.append(nc)

        title = data.get("title")
        return NotebookResult(
            path=path,
            cells=cells,
            backend=self.name,
            title=title,
        )


# ---------------------------------------------------------------------------
# Capability-based dispatch
# ---------------------------------------------------------------------------

# Singleton backends
_python_backend = PythonSyntaxBackend()
_kernel_backend = KernelSemanticBackend()


def get_backends_for_capability(capability: str) -> list:
    """Return ordered list of backends suitable for the given capability.

    Dispatch by task, not by transport:
    - outline/text: Python parser first — fast, offline, sufficient for
      structural or plain-text reads.  Kernel fallback only if Python fails.
    - code/semantic: Kernel preferred for accuracy, Python fallback
    - structure: Kernel preferred, Python fallback
    - full: All cells — prefer kernel, fall back to Python
    - live: Addon only (not handled here — separate code path in server.py)
    """
    if capability in ("outline", "text"):
        return [_python_backend, _kernel_backend]
    elif capability in ("code", "semantic") or capability == "structure":
        return [_kernel_backend, _python_backend]
    else:  # "full" or unrecognized
        return [_kernel_backend, _python_backend]


async def extract_notebook(
    path: str,
    *,
    capability: str = "full",
    cell_types: list[str] | None = None,
    view: CellView = CellView.SEMANTIC,
    include_alternates: bool = False,
    truncation_threshold: int = 25000,
    force_backend: str | None = None,
) -> NotebookResult:
    """Extract notebook content using capability-based dispatch.

    Args:
        path: Path to the .nb file
        capability: What the caller needs — "outline", "text", "code",
                    "semantic", "full", "structure"
        cell_types: Optional list of cell styles to include
        view: Which view to use as the primary content
        include_alternates: If True, include alternate views in result
        truncation_threshold: Max chars per cell (0 to disable)
        force_backend: Override dispatch — "python_syntax" or "kernel_semantic"
    """
    if force_backend == "python_syntax":
        backends = [_python_backend]
    elif force_backend == "kernel_semantic":
        backends = [_kernel_backend]
    elif view == CellView.RAW:
        # Raw view needs BoxData which only the Python parser preserves.
        # Prefer Python; kernel fallback will mark cells as lossy.
        backends = [_python_backend, _kernel_backend]
    else:
        backends = get_backends_for_capability(capability)

    last_error: Exception | None = None
    for backend in backends:
        if not backend.available():
            continue
        try:
            return await backend.extract(
                path,
                cell_types=cell_types,
                view=view,
                include_alternates=include_alternates,
                truncation_threshold=truncation_threshold,
            )
        except Exception as e:
            logger.warning("Backend %s failed for %s: %s", backend.name, path, e)
            last_error = e

    # All backends failed
    if last_error:
        return NotebookResult(
            path=path,
            cells=[],
            backend="none",
            error=f"All backends failed. Last error: {last_error}",
        )
    return NotebookResult(
        path=path,
        cells=[],
        backend="none",
        error="No available backend",
    )
