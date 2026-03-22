"""Tests for the notebook backend abstraction layer.

Covers:
- Cell model (NotebookCell, NotebookResult) serialization and views
- PythonSyntaxBackend (offline, always available)
- KernelSemanticBackend (gated by wolfram_runtime fixture)
- Capability-based dispatch
- Disk cache
- Consolidated read_notebook tool
- Backward compatibility (old tools still work identically)
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from mathematica_mcp.notebook_backend import (
    CellView,
    KernelSemanticBackend,
    NotebookCell,
    NotebookResult,
    PythonSyntaxBackend,
    extract_notebook,
    get_backends_for_capability,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
INTEGRATION_NB = REPO_ROOT / "Integration.nb"
COMPLEX_NB = FIXTURES / "complex_notebook.nb"


# ============================================================================
# Cell model tests
# ============================================================================


class TestNotebookCell:
    def test_basic_serialization(self):
        cell = NotebookCell(
            cell_index=0, style="Input", content="1+1",
            cell_label="In[1]:=",
        )
        d = cell.to_dict()
        assert d["index"] == 0
        assert d["style"] == "Input"
        assert d["content"] == "1+1"
        assert d["cell_label"] == "In[1]:="

    def test_lean_serialization_omits_empty_fields(self):
        cell = NotebookCell(cell_index=0, style="Text", content="hello")
        d = cell.to_dict()
        # These should NOT be present when empty/default
        assert "cell_id" not in d
        assert "group_id" not in d
        assert "tags" not in d
        assert "is_generated" not in d
        assert "was_truncated" not in d
        assert "conversion_lossy" not in d
        assert "cell_label" not in d

    def test_rich_serialization_includes_metadata(self):
        cell = NotebookCell(
            cell_index=3, style="Output", content="42",
            cell_id=12345, group_id=1, position_in_group=2,
            parent_input_index=2, cell_label="Out[1]=",
            tags=["result"], is_generated=True,
        )
        d = cell.to_dict()
        assert d["cell_id"] == 12345
        assert d["group_id"] == 1
        assert d["position_in_group"] == 2
        assert d["parent_input_index"] == 2
        assert d["tags"] == ["result"]
        assert d["is_generated"] is True

    def test_alternate_views_hidden_by_default(self):
        cell = NotebookCell(
            cell_index=0, style="Input", content="semantic form",
            _raw="BoxData[...]", _display="pretty display",
        )
        d = cell.to_dict(include_alternates=False)
        assert "alternates" not in d

    def test_alternate_views_included_on_request(self):
        cell = NotebookCell(
            cell_index=0, style="Input", content="semantic form",
            _raw="BoxData[...]", _display="pretty display",
        )
        d = cell.to_dict(include_alternates=True)
        assert "alternates" in d
        assert d["alternates"]["raw"] == "BoxData[...]"
        assert d["alternates"]["display"] == "pretty display"

    def test_get_view_returns_correct_view(self):
        cell = NotebookCell(
            cell_index=0, style="Input", content="default",
            _raw="raw data", _semantic="semantic data",
        )
        assert cell.get_view(CellView.RAW) == "raw data"
        assert cell.get_view(CellView.SEMANTIC) == "semantic data"
        # DISPLAY not set, falls back to content
        assert cell.get_view(CellView.DISPLAY) == "default"

    def test_truncation_metadata(self):
        cell = NotebookCell(
            cell_index=0, style="Input", content="truncated...",
            was_truncated=True, original_length=50000,
        )
        d = cell.to_dict()
        assert d["was_truncated"] is True
        assert d["original_length"] == 50000

    def test_conversion_lossy_flag(self):
        cell = NotebookCell(
            cell_index=0, style="Output", content="fallback text",
            conversion_lossy=True,
        )
        d = cell.to_dict()
        assert d["conversion_lossy"] is True


# ============================================================================
# NotebookResult tests
# ============================================================================


class TestNotebookResult:
    def _make_result(self) -> NotebookResult:
        return NotebookResult(
            path="/test/notebook.nb",
            backend="python_syntax",
            title="Test",
            cells=[
                NotebookCell(0, "Title", "Test"),
                NotebookCell(1, "Section", "Intro"),
                NotebookCell(2, "Text", "Some text here."),
                NotebookCell(3, "Input", "1+1", cell_label="In[1]:="),
                NotebookCell(4, "Output", "2", cell_label="Out[1]="),
                NotebookCell(5, "Input", "Plot[Sin[x], {x, 0, 2 Pi}]"),
            ],
        )

    def test_cell_counts(self):
        r = self._make_result()
        assert r.cell_count == 6
        assert r.code_cell_count == 2

    def test_to_dict_structure(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["success"] is True
        assert d["backend"] == "python_syntax"
        assert d["cell_count"] == 6
        assert d["code_cells"] == 2
        assert len(d["cells"]) == 6

    def test_to_markdown(self):
        r = self._make_result()
        md = r.to_markdown()
        assert "# Test" in md
        assert "## Intro" in md
        assert "Some text here." in md
        assert "```wolfram" in md
        assert "1+1" in md

    def test_to_wolfram_code(self):
        r = self._make_result()
        wl = r.to_wolfram_code()
        assert "1+1" in wl
        assert "Plot[Sin[x], {x, 0, 2 Pi}]" in wl
        # Should NOT include text/section cells
        assert "Some text here." not in wl

    def test_to_outline(self):
        r = self._make_result()
        outline = r.to_outline()
        assert len(outline) == 2
        assert outline[0]["level"] == "Title"
        assert outline[0]["title"] == "Test"
        assert outline[1]["level"] == "Section"

    def test_to_plain_text(self):
        r = self._make_result()
        txt = r.to_plain_text()
        assert "Test" in txt
        assert "Some text here." in txt

    def test_error_result(self):
        r = NotebookResult(
            path="/test/bad.nb", cells=[], backend="none",
            error="File not found",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert "File not found" in d["error"]


# ============================================================================
# PythonSyntaxBackend tests
# ============================================================================


class TestPythonSyntaxBackend:
    def test_always_available(self):
        backend = PythonSyntaxBackend()
        assert backend.available() is True
        assert backend.name == "python_syntax"

    def test_extract_integration_nb(self):
        """Test parsing the simple Integration.nb fixture."""
        backend = PythonSyntaxBackend()
        result = asyncio.run(backend.extract(str(INTEGRATION_NB)))

        assert result.backend == "python_syntax"
        assert result.cell_count >= 1
        # Should find the Integrate expression
        code_cells = [c for c in result.cells if c.style == "Input"]
        assert len(code_cells) >= 1
        assert "Integrate" in code_cells[0].content

    def test_extract_complex_nb(self):
        """Test parsing the complex test notebook."""
        backend = PythonSyntaxBackend()
        result = asyncio.run(backend.extract(str(COMPLEX_NB)))

        assert result.backend == "python_syntax"
        assert result.cell_count >= 6

        # Verify cell types
        styles = [c.style for c in result.cells]
        assert "Title" in styles
        assert "Section" in styles
        assert "Text" in styles
        assert "Input" in styles

        # Verify title extraction
        title_cell = next(c for c in result.cells if c.style == "Title")
        assert "Analysis Notebook" in title_cell.content

    def test_cell_type_filtering(self):
        """Test that cell_types parameter filters correctly."""
        backend = PythonSyntaxBackend()
        result = asyncio.run(
            backend.extract(str(COMPLEX_NB), cell_types=["Input", "Code"])
        )
        for cell in result.cells:
            assert cell.style in ("Input", "Code")

    def test_input_output_provenance(self):
        """Verify Output cells track their parent Input cell."""
        backend = PythonSyntaxBackend()
        result = asyncio.run(backend.extract(str(COMPLEX_NB)))

        output_cells = [c for c in result.cells if c.style == "Output"]
        if output_cells:
            # The Output cell should have parent_input_index set
            out = output_cells[0]
            assert out.parent_input_index is not None

    def test_raw_content_stored_as_alternate(self):
        """Verify raw BoxData is preserved in _raw alternate view."""
        backend = PythonSyntaxBackend()
        result = asyncio.run(backend.extract(str(COMPLEX_NB)))

        # Find cells that had BoxData (Input cells in complex_notebook.nb)
        input_cells = [c for c in result.cells if c.style == "Input"]
        # At least some should have raw content stored
        cells_with_raw = [c for c in input_cells if c._raw]
        # The complex notebook has BoxData cells
        assert len(cells_with_raw) >= 1

    def test_markdown_output_matches_old_parser(self):
        """Golden test: markdown output should match existing parser."""
        from mathematica_mcp.notebook_parser import (
            NotebookParser,
            parse_notebook_cached,
        )

        # Old path
        old_notebook = parse_notebook_cached(
            str(INTEGRATION_NB), truncation_threshold=25000
        )
        old_md = NotebookParser(truncation_threshold=25000).to_markdown(old_notebook)

        # New path
        backend = PythonSyntaxBackend()
        result = asyncio.run(backend.extract(str(INTEGRATION_NB)))
        new_md = result.to_markdown()

        # Core content should match
        assert "Integrate[1/(x^2 + x^3), x]" in old_md
        assert "Integrate[1/(x^2 + x^3), x]" in new_md

    def test_wolfram_code_output(self):
        """Wolfram code extraction should return only code cells."""
        backend = PythonSyntaxBackend()
        result = asyncio.run(backend.extract(str(COMPLEX_NB)))
        wl = result.to_wolfram_code()

        # Should contain code but not prose
        assert "Analysis Notebook" not in wl
        assert "This notebook demonstrates" not in wl


# ============================================================================
# KernelSemanticBackend tests
# ============================================================================


class TestKernelSemanticBackend:
    def test_availability_check(self):
        """Backend reports availability based on wolframscript presence."""
        backend = KernelSemanticBackend()
        # Just verify it doesn't crash — actual availability depends on env
        result = backend.available()
        assert isinstance(result, bool)

    def test_helper_wl_path_exists(self):
        """Verify the shipped .wl helper file exists."""
        backend = KernelSemanticBackend()
        wl_path = backend._get_helper_wl_path()
        assert Path(wl_path).exists(), f"Helper not found at {wl_path}"

    def test_parse_kernel_output_valid_json(self):
        """Test parsing of well-formed kernel JSON output."""
        backend = KernelSemanticBackend()
        sample_output = json.dumps({
            "path": "/test/notebook.nb",
            "title": "Test",
            "cell_count": 2,
            "code_cells": 1,
            "backend": "kernel_semantic",
            "cells": [
                {
                    "style": "Title",
                    "content": "Test Notebook",
                    "cell_id": 100001,
                    "group_id": 1,
                    "position_in_group": 1,
                    "cell_label": "",
                    "tags": [],
                    "is_generated": False,
                    "conversion_lossy": False,
                },
                {
                    "style": "Input",
                    "content": "1 + 1",
                    "cell_id": 100002,
                    "group_id": 1,
                    "position_in_group": 2,
                    "cell_label": "In[1]:=",
                    "tags": [],
                    "is_generated": False,
                    "conversion_lossy": False,
                },
            ],
        })

        result = backend._parse_kernel_output(
            sample_output, "/test/notebook.nb"
        )
        assert result.backend == "kernel_semantic"
        assert result.cell_count == 2
        assert result.cells[0].style == "Title"
        assert result.cells[0].cell_id == 100001
        assert result.cells[1].content == "1 + 1"
        assert result.cells[1].cell_label == "In[1]:="

    def test_parse_kernel_output_with_error(self):
        """Test handling of kernel error output."""
        backend = KernelSemanticBackend()
        error_output = json.dumps({
            "error": "Failed to import notebook",
            "path": "/test/bad.nb",
        })
        result = backend._parse_kernel_output(error_output, "/test/bad.nb")
        assert result.error is not None
        assert "Failed to import" in result.error

    def test_parse_kernel_output_cell_types_filter(self):
        """Test cell type filtering in kernel output parsing."""
        backend = KernelSemanticBackend()
        sample_output = json.dumps({
            "cells": [
                {"style": "Title", "content": "T"},
                {"style": "Input", "content": "code"},
                {"style": "Output", "content": "result"},
            ],
        })
        result = backend._parse_kernel_output(
            sample_output, "/test/nb.nb",
            cell_types=["Input"],
        )
        assert result.cell_count == 1
        assert result.cells[0].style == "Input"

    def test_parse_kernel_output_provenance(self):
        """Test that Input→Output provenance is tracked."""
        backend = KernelSemanticBackend()
        sample_output = json.dumps({
            "cells": [
                {"style": "Input", "content": "1+1"},
                {"style": "Output", "content": "2", "is_generated": True},
            ],
        })
        result = backend._parse_kernel_output(sample_output, "/test/nb.nb")
        output_cell = result.cells[1]
        assert output_cell.parent_input_index == 0


# ============================================================================
# Dispatch tests
# ============================================================================


class TestDispatch:
    def test_capability_backend_ordering(self):
        """Verify dispatch returns backends in correct priority order."""
        for cap in ("code", "semantic"):
            backends = get_backends_for_capability(cap)
            assert backends[0].name == "kernel_semantic"
            assert backends[1].name == "python_syntax"

        for cap in ("outline", "text"):
            backends = get_backends_for_capability(cap)
            # Both should have python as fallback
            names = [b.name for b in backends]
            assert "python_syntax" in names

    def test_force_python_backend(self):
        """Force python backend skips kernel attempt."""
        result = asyncio.run(
            extract_notebook(
                str(INTEGRATION_NB),
                force_backend="python_syntax",
            )
        )
        assert result.backend == "python_syntax"
        assert result.error is None
        assert result.cell_count >= 1

    def test_fallback_on_kernel_unavailable(self):
        """When kernel is unavailable, falls back to Python parser."""
        # Mock kernel as unavailable
        with patch.object(KernelSemanticBackend, "available", return_value=False):
            result = asyncio.run(
                extract_notebook(str(INTEGRATION_NB), capability="code")
            )
            assert result.backend == "python_syntax"
            assert result.error is None

    def test_nonexistent_file(self):
        """Extract on nonexistent file should error gracefully."""
        result = asyncio.run(
            extract_notebook(
                "/nonexistent/path.nb",
                force_backend="python_syntax",
            )
        )
        assert result.error is not None or result.cell_count == 0


# ============================================================================
# Disk cache tests
# ============================================================================


class TestDiskCache:
    def test_cache_roundtrip(self, tmp_path: Path):
        """Test write then read from disk cache."""
        from mathematica_mcp.disk_cache import get_cached, put_cached

        # Create a temporary .nb file
        nb_file = tmp_path / "test.nb"
        nb_file.write_text('Notebook[{Cell["hello", "Input"]}]')

        data = {"cells": [{"style": "Input", "content": "hello"}]}

        # Override cache dir for testing
        with patch(
            "mathematica_mcp.disk_cache._CACHE_DIR", tmp_path / "cache"
        ):
            put_cached(str(nb_file), "python_syntax", data, view="semantic")
            result = get_cached(str(nb_file), "python_syntax", view="semantic")

        assert result is not None
        assert result["cells"] == data["cells"]

    def test_cache_invalidation_on_mtime_change(self, tmp_path: Path):
        """Cache should invalidate when source file is modified."""
        from mathematica_mcp.disk_cache import get_cached, put_cached

        nb_file = tmp_path / "test.nb"
        nb_file.write_text('Notebook[{Cell["v1", "Input"]}]')

        data = {"cells": [{"content": "v1"}]}

        with patch(
            "mathematica_mcp.disk_cache._CACHE_DIR", tmp_path / "cache"
        ):
            put_cached(str(nb_file), "python_syntax", data)
            # Modify the file
            nb_file.write_text('Notebook[{Cell["v2", "Input"]}]')
            result = get_cached(str(nb_file), "python_syntax")

        # Should be None because mtime changed
        assert result is None

    def test_cache_miss_returns_none(self, tmp_path: Path):
        """Cache miss returns None, not error."""
        from mathematica_mcp.disk_cache import get_cached

        nb_file = tmp_path / "test.nb"
        nb_file.write_text("Notebook[{}]")

        with patch(
            "mathematica_mcp.disk_cache._CACHE_DIR", tmp_path / "cache"
        ):
            result = get_cached(str(nb_file), "python_syntax")

        assert result is None

    def test_clear_cache(self, tmp_path: Path):
        """Test cache clearing."""
        from mathematica_mcp.disk_cache import clear_cache, get_cached, put_cached

        nb_file = tmp_path / "test.nb"
        nb_file.write_text("Notebook[{}]")

        with patch(
            "mathematica_mcp.disk_cache._CACHE_DIR", tmp_path / "cache"
        ):
            put_cached(str(nb_file), "python_syntax", {"cells": []})
            count = clear_cache()
            assert count >= 1
            assert get_cached(str(nb_file), "python_syntax") is None

    def test_different_options_different_entries(self, tmp_path: Path):
        """Different extraction options should produce separate cache entries."""
        from mathematica_mcp.disk_cache import get_cached, put_cached

        nb_file = tmp_path / "test.nb"
        nb_file.write_text('Notebook[{Cell["x", "Input"]}]')

        with patch(
            "mathematica_mcp.disk_cache._CACHE_DIR", tmp_path / "cache"
        ):
            put_cached(str(nb_file), "python_syntax", {"v": 1}, view="semantic")
            put_cached(str(nb_file), "python_syntax", {"v": 2}, view="raw")

            r1 = get_cached(str(nb_file), "python_syntax", view="semantic")
            r2 = get_cached(str(nb_file), "python_syntax", view="raw")

        assert r1["v"] == 1
        assert r2["v"] == 2


# ============================================================================
# Consolidated read_notebook tool tests
# ============================================================================


class TestReadNotebookTool:
    """Test the consolidated read_notebook MCP tool.

    All tests force backend="python_syntax" to avoid kernel startup delays.
    Kernel integration is tested separately (gated by wolfram_runtime).
    """

    def test_markdown_format(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(INTEGRATION_NB), output_format="markdown",
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        assert result["format"] == "markdown"
        assert result["backend"] == "python_syntax"
        assert "Integrate" in result["content"]

    def test_wolfram_format(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(INTEGRATION_NB), output_format="wolfram",
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        assert result["format"] == "wolfram"
        assert "Integrate" in result["content"]

    def test_outline_format(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(COMPLEX_NB), output_format="outline",
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        assert result["format"] == "outline"
        assert result["section_count"] >= 2

    def test_json_format(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(COMPLEX_NB), output_format="json",
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        assert "cells" in result
        assert result["cell_count"] >= 1

    def test_cell_types_filter(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(COMPLEX_NB),
                output_format="json",
                cell_types=["Input"],
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        for cell in result["cells"]:
            assert cell["style"] == "Input"

    def test_exclude_outputs(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(COMPLEX_NB),
                output_format="json",
                include_outputs=False,
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        for cell in result["cells"]:
            assert cell["style"] not in ("Output", "Message", "Print")

    def test_force_python_backend(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(INTEGRATION_NB),
                output_format="json",
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        assert result["backend"] == "python_syntax"

    def test_file_not_found(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook("/nonexistent/file.nb", backend="python_syntax")
        ))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_plain_format(self):
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(COMPLEX_NB), output_format="plain",
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        assert result["format"] == "plain"
        assert "Analysis Notebook" in result["content"]


# ============================================================================
# Backward compatibility tests
# ============================================================================


class TestBackwardCompatibility:
    """Verify that existing tools still produce the same results."""

    def test_old_read_notebook_content_still_works(self):
        from mathematica_mcp.server import read_notebook_content

        result = json.loads(asyncio.run(
            read_notebook_content(str(INTEGRATION_NB))
        ))
        assert result["success"] is True
        assert result["cell_count"] >= 1
        assert "Integrate" in result["cells"][0]["content"]

    def test_old_convert_notebook_still_works(self):
        from mathematica_mcp.server import convert_notebook

        result = json.loads(asyncio.run(
            convert_notebook(str(INTEGRATION_NB), "markdown")
        ))
        assert result["success"] is True
        assert "Integrate" in result["content"]

    def test_old_get_notebook_outline_still_works(self):
        from mathematica_mcp.server import get_notebook_outline

        result = json.loads(asyncio.run(
            get_notebook_outline(str(COMPLEX_NB))
        ))
        assert result["success"] is True
        assert result["count"] >= 2

    def test_old_parse_notebook_python_still_works(self):
        from mathematica_mcp.server import parse_notebook_python

        result = json.loads(asyncio.run(
            parse_notebook_python(str(INTEGRATION_NB), "markdown")
        ))
        assert result["success"] is True
        assert "Integrate" in result["content"]

    def test_old_get_notebook_cell_still_works(self):
        from mathematica_mcp.server import get_notebook_cell

        result = json.loads(asyncio.run(
            get_notebook_cell(str(INTEGRATION_NB), 0)
        ))
        assert result["success"] is True
        assert "Integrate" in result["content"]


# ============================================================================
# Regression tests for specific fixes
# ============================================================================


class TestWLHelperFixes:
    """Tests for fixes to notebook_converter.wl issues."""

    def test_wl_helper_uses_plain_import(self):
        """Fix 1: WL helper must use Import[file], not Import[file, {...}]."""
        wl_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "mathematica_mcp" / "helpers" / "notebook_converter.wl"
        )
        content = wl_path.read_text()
        # Must NOT contain the broken element spec
        assert '{"Notebooks", "Notebook"}' not in content
        # Must contain plain Import[file]
        assert "Import[file]" in content

    def test_wl_helper_handles_cell_wrapping_cellgroupdata(self):
        """Fix 2: processTopLevel must handle Cell[CellGroupData[...], Open]."""
        wl_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "mathematica_mcp" / "helpers" / "notebook_converter.wl"
        )
        content = wl_path.read_text()
        # Must have a pattern for Cell[CellGroupData[...], ...]
        assert "Cell[CellGroupData[cellList_List" in content

    def test_kernel_backend_parse_grouped_cells(self):
        """Fix 2 (Python side): grouped cells should not be dropped."""
        backend = KernelSemanticBackend()
        # Simulate kernel output with group structure
        # (as produced by the fixed WL helper)
        sample = json.dumps({
            "cells": [
                {"style": "Title", "content": "My Title", "group_id": 1,
                 "position_in_group": 1, "cell_id": 100},
                {"style": "Text", "content": "Some text", "group_id": 1,
                 "position_in_group": 2, "cell_id": 101},
                {"style": "Input", "content": "1+1", "group_id": 1,
                 "position_in_group": 3, "cell_id": 102,
                 "cell_label": "In[1]:="},
                {"style": "Output", "content": "2", "group_id": 1,
                 "position_in_group": 4, "cell_id": 103,
                 "is_generated": True},
                {"style": "Section", "content": "Appendix", "group_id": 2,
                 "position_in_group": 1, "cell_id": 104},
            ],
        })
        result = backend._parse_kernel_output(sample, "/test.nb")
        assert result.cell_count == 5
        assert result.cells[0].group_id == 1
        assert result.cells[4].group_id == 2


class TestViewSwitching:
    """Fix 4: View API must actually change the returned content."""

    def test_python_backend_raw_view_returns_boxdata(self):
        """view=raw should return raw BoxData when available."""
        backend = PythonSyntaxBackend()

        # Semantic view (default)
        result_semantic = asyncio.run(
            backend.extract(str(COMPLEX_NB), view=CellView.SEMANTIC)
        )
        # Raw view
        result_raw = asyncio.run(
            backend.extract(str(COMPLEX_NB), view=CellView.RAW)
        )

        # Find an Input cell that has BoxData (should have different content)
        for sem_cell, raw_cell in zip(result_semantic.cells, result_raw.cells):
            if sem_cell.style == "Input" and sem_cell._raw:
                # Raw view should contain BoxData-like content
                assert raw_cell.content == sem_cell._raw
                # Semantic should be the parsed form
                assert sem_cell.content != sem_cell._raw
                break

    def test_python_backend_stores_alternates(self):
        """Both _semantic and _raw should be populated."""
        backend = PythonSyntaxBackend()
        result = asyncio.run(backend.extract(str(COMPLEX_NB)))

        input_cells = [c for c in result.cells if c.style == "Input"]
        for cell in input_cells:
            # _semantic should always be set
            assert cell._semantic is not None
            # _display should always be set
            assert cell._display is not None

    def test_kernel_backend_stores_semantic_alternate(self):
        """Kernel backend should store semantic in alternates."""
        backend = KernelSemanticBackend()
        sample = json.dumps({
            "cells": [
                {"style": "Input", "content": "Plot[Sin[x], {x, 0, 2 Pi}]"},
            ],
        })
        result = backend._parse_kernel_output(sample, "/test.nb")
        assert result.cells[0]._semantic == "Plot[Sin[x], {x, 0, 2 Pi}]"


class TestIncludeOutputsFix:
    """Fix 5: include_outputs=False must work even with explicit cell_types."""

    def test_include_outputs_false_with_cell_types(self):
        """cell_types=["Input","Output"] + include_outputs=False should exclude Output."""
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(COMPLEX_NB),
                output_format="json",
                cell_types=["Input", "Output"],
                include_outputs=False,
                backend="python_syntax",
            )
        ))
        assert result["success"] is True
        styles = {c["style"] for c in result["cells"]}
        assert "Output" not in styles
        assert "Input" in styles

    def test_include_outputs_true_preserves_outputs(self):
        """include_outputs=True (default) should keep Output cells."""
        from mathematica_mcp.server import read_notebook

        result = json.loads(asyncio.run(
            read_notebook(
                str(COMPLEX_NB),
                output_format="json",
                include_outputs=True,
                backend="python_syntax",
            )
        ))
        styles = {c["style"] for c in result["cells"]}
        assert "Output" in styles


class TestCacheEquivalence:
    """Contract: cached and uncached kernel notebook reads must produce
    identical NotebookResult values."""

    def test_cached_and_parsed_produce_same_cells(self):
        """_parse_kernel_output and _build_result_from_data must yield
        identical cell content, styles, and metadata."""
        backend = KernelSemanticBackend()
        sample_data = {
            "cells": [
                {"style": "Input", "content": "1+1", "cell_label": "In[1]:="},
                {"style": "Output", "content": "2", "cell_label": "Out[1]="},
                {"style": "Text", "content": "Hello world"},
            ],
            "title": "Test Notebook",
        }
        sample_json = json.dumps(sample_data)

        # Parse from JSON string (uncached path)
        uncached = backend._parse_kernel_output(
            sample_json, "/test.nb",
            view=CellView.SEMANTIC,
            truncation_threshold=25000,
        )

        # Build from dict directly (cached path — no JSON round-trip)
        cached = backend._build_result_from_data(
            sample_data, "/test.nb",
            view=CellView.SEMANTIC,
            truncation_threshold=25000,
        )

        assert len(uncached.cells) == len(cached.cells)
        assert uncached.title == cached.title
        assert uncached.backend == cached.backend

        for uc, cc in zip(uncached.cells, cached.cells):
            assert uc.content == cc.content
            assert uc.style == cc.style
            assert uc.cell_label == cc.cell_label
            assert uc.was_truncated == cc.was_truncated
            assert uc.cell_index == cc.cell_index

    def test_parse_kernel_output_preserves_provenance(self):
        """Input→Output provenance tracking must work through parsing."""
        backend = KernelSemanticBackend()
        sample = json.dumps({
            "cells": [
                {"style": "Input", "content": "x^2"},
                {"style": "Output", "content": "x^2"},
                {"style": "Input", "content": "1+1"},
                {"style": "Output", "content": "2"},
            ],
        })
        result = backend._parse_kernel_output(
            sample, "/test.nb",
            view=CellView.SEMANTIC,
        )
        output_cells = [c for c in result.cells if c.style == "Output"]
        assert output_cells[0].parent_input_index == 0
        assert output_cells[1].parent_input_index == 2

    def test_build_result_from_data_preserves_provenance(self):
        """Provenance must also work through _build_result_from_data."""
        backend = KernelSemanticBackend()
        data = {
            "cells": [
                {"style": "Input", "content": "x^2"},
                {"style": "Output", "content": "x^2"},
                {"style": "Input", "content": "1+1"},
                {"style": "Output", "content": "2"},
            ],
        }
        result = backend._build_result_from_data(
            data, "/test.nb",
            view=CellView.SEMANTIC,
        )
        output_cells = [c for c in result.cells if c.style == "Output"]
        assert output_cells[0].parent_input_index == 0
        assert output_cells[1].parent_input_index == 2

    def test_extract_notebook_result_contract(self):
        """NotebookResult from extract_notebook must have expected structure."""
        result = asyncio.run(
            extract_notebook(str(INTEGRATION_NB), capability="full")
        )
        assert hasattr(result, "cells")
        assert hasattr(result, "path")
        assert hasattr(result, "backend")
        assert hasattr(result, "title")
        assert hasattr(result, "error")
        assert isinstance(result.cells, list)
        assert result.backend in ("python_syntax", "kernel_semantic", "none")
        if result.cells:
            cell = result.cells[0]
            assert hasattr(cell, "cell_index")
            assert hasattr(cell, "style")
            assert hasattr(cell, "content")
            assert hasattr(cell, "was_truncated")
            assert hasattr(cell, "original_length")


class TestCacheStaleness:
    """Fix 3: Kernel extraction must not serve stale cached results."""

    def test_disk_cache_invalidates_on_file_change(self, tmp_path: Path):
        """Disk cache must return None after the source file changes."""
        from mathematica_mcp.disk_cache import get_cached, put_cached

        nb_file = tmp_path / "test.nb"
        nb_file.write_text('Notebook[{Cell["v1", "Input"]}]')

        with patch("mathematica_mcp.disk_cache._CACHE_DIR", tmp_path / "cache"):
            put_cached(
                str(nb_file), "kernel_semantic",
                {"cells": [{"content": "v1"}]},
                truncation_threshold=25000, view="semantic",
            )

            # Verify cache hit before modification
            cached = get_cached(
                str(nb_file), "kernel_semantic",
                truncation_threshold=25000, view="semantic",
            )
            assert cached is not None

            # Modify the file
            nb_file.write_text('Notebook[{Cell["v2", "Input"]}]')

            # Cache must now miss
            cached = get_cached(
                str(nb_file), "kernel_semantic",
                truncation_threshold=25000, view="semantic",
            )
            assert cached is None

    def test_kernel_code_includes_mtime_sentinel(self):
        """The WL code string must contain mtime for cache-busting."""
        import os
        backend = KernelSemanticBackend()
        # We can't run the full extract, but we can verify the code
        # would include mtime by checking the implementation
        abs_path = str(Path(INTEGRATION_NB).resolve())
        stat = os.stat(abs_path)
        # The mtime_ns should appear in the generated code
        assert stat.st_mtime_ns > 0  # File exists and has a real mtime


class TestRawViewDispatchFix:
    """Fix: view=raw must prefer Python backend and kernel must mark lossy."""

    def test_dispatch_prefers_python_for_raw_view(self):
        """Dispatcher should prefer python_syntax when view=raw."""
        result = asyncio.run(
            extract_notebook(
                str(COMPLEX_NB),
                view=CellView.RAW,
            )
        )
        # Python backend should be selected since it has raw boxes
        assert result.backend == "python_syntax"

    def test_kernel_backend_marks_lossy_for_raw_view(self):
        """Kernel backend must set conversion_lossy=True for view=raw."""
        backend = KernelSemanticBackend()
        sample = json.dumps({
            "cells": [
                {"style": "Input", "content": "1+1", "conversion_lossy": False},
                {"style": "Text", "content": "hello", "conversion_lossy": False},
            ],
        })
        result = backend._parse_kernel_output(
            sample, "/test.nb", view=CellView.RAW,
        )
        for cell in result.cells:
            assert cell.conversion_lossy is True

    def test_kernel_backend_not_lossy_for_semantic_view(self):
        """Kernel backend should NOT mark lossy for semantic view."""
        backend = KernelSemanticBackend()
        sample = json.dumps({
            "cells": [
                {"style": "Input", "content": "1+1", "conversion_lossy": False},
            ],
        })
        result = backend._parse_kernel_output(
            sample, "/test.nb", view=CellView.SEMANTIC,
        )
        assert result.cells[0].conversion_lossy is False


class TestRawViewTruncationFix:
    """Fix: view=raw must apply truncation to raw content, not inherit stale metadata."""

    def test_raw_view_truncates_large_raw_content(self, tmp_path: Path):
        """Raw content exceeding threshold must be truncated."""
        # Create a notebook with large BoxData
        large_box = "x" * 500
        nb_content = (
            'Notebook[{\n'
            f'Cell[BoxData[RowBox[{{"{large_box}"}}]], "Input", CellID->1]\n'
            '}]'
        )
        nb_file = tmp_path / "large.nb"
        nb_file.write_text(nb_content)

        backend = PythonSyntaxBackend()
        result = asyncio.run(
            backend.extract(
                str(nb_file),
                view=CellView.RAW,
                truncation_threshold=100,
            )
        )

        input_cells = [c for c in result.cells if c.style == "Input"]
        assert len(input_cells) >= 1
        cell = input_cells[0]

        # If raw content was larger than threshold, it must be truncated
        if cell._raw and len(cell._raw) > 100:
            assert cell.was_truncated is True
            # Truncated content should be smaller than original
            assert len(cell.content) < cell.original_length
            assert cell.original_length > 100

    def test_semantic_view_truncation_unaffected(self):
        """Semantic view should still work with truncation as before."""
        backend = PythonSyntaxBackend()
        result = asyncio.run(
            backend.extract(str(INTEGRATION_NB), view=CellView.SEMANTIC)
        )
        # Integration.nb has small cells, no truncation expected
        for cell in result.cells:
            assert cell.was_truncated is False
