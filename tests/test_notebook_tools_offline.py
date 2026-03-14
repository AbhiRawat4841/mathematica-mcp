from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mathematica_mcp.notebook_parser import (
    NotebookParser,
    _parse_notebook_cached,
    convert_special_chars,
    parse_notebook_cached,
)
from mathematica_mcp.server import (
    convert_notebook,
    get_notebook_outline,
    read_notebook_content,
)


def test_convert_special_chars_translates_known_tokens():
    text = r"\[Alpha] + \[Rule] + \[Infinity]"
    assert convert_special_chars(text) == "α + → + ∞"


def test_parse_notebook_cached_reuses_parsed_structure(monkeypatch):
    calls = {"count": 0}
    original = NotebookParser.parse_file
    _parse_notebook_cached.cache_clear()

    def wrapped(self, path):
        calls["count"] += 1
        return original(self, path)

    monkeypatch.setattr(NotebookParser, "parse_file", wrapped)

    first = parse_notebook_cached("Integration.nb", truncation_threshold=25000)
    second = parse_notebook_cached("Integration.nb", truncation_threshold=25000)

    assert calls["count"] == 1
    assert first is second


def test_read_notebook_content_uses_python_parser():
    result = json.loads(asyncio.run(read_notebook_content("Integration.nb")))

    assert result["success"] is True
    assert result["cell_count"] == 1
    assert result["cells"][0]["style"] == "Input"
    assert "Integrate[1/(x^2 + x^3), x]" in result["cells"][0]["content"]


def test_convert_notebook_markdown_avoids_subprocess(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("subprocess path should not be used for markdown")

    monkeypatch.setattr("subprocess.run", explode)

    result = json.loads(asyncio.run(convert_notebook("Integration.nb", "markdown")))

    assert result["success"] is True
    assert "```wolfram" in result["content"]
    assert "Integrate[1/(x^2 + x^3), x]" in result["content"]


def test_get_notebook_outline_uses_parser_for_sections(tmp_path: Path):
    notebook_path = tmp_path / "outline.nb"
    notebook_path.write_text(
        '\n'.join(
            [
                "Notebook[{",
                'Cell["Doc Title", "Title"],',
                'Cell["Intro", "Section"],',
                'Cell["Detail", "Subsection"],',
                'Cell["1+1", "Input"]',
                "}]",
            ]
        ),
        encoding="utf-8",
    )

    result = json.loads(asyncio.run(get_notebook_outline(str(notebook_path))))

    assert result["success"] is True
    assert result["count"] == 3
    assert result["sections"][0]["title"] == "Doc Title"
    assert result["sections"][1]["level"] == "Section"
