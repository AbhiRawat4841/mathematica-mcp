"""Normalize MCP tool responses into a uniform shape for verification.

Handles: JSON strings, dicts, lists, plain strings, Image objects,
and parse_error=true payloads with raw content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Artifact:
    """A produced artifact — file on disk or in-memory image."""

    kind: Literal["file", "image", "bytes"]
    path: str | None = None
    data: bytes | None = None
    format: str | None = None


@dataclass
class NormalizedResult:
    """Uniform result shape consumed by all verifiers."""

    ok: bool = False
    parsed: Any | None = None
    raw: str | None = None
    output_text: str | None = None
    warnings: list[str] = field(default_factory=list)
    error_text: str | None = None
    parse_error: bool = False
    artifacts: list[Artifact] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def normalize(response: Any, tool_name: str | None = None) -> NormalizedResult:
    """Normalize any tool response into a NormalizedResult."""
    # Image objects (rasterize_expression, screenshot_notebook, screenshot_cell)
    try:
        from mcp.server.fastmcp import Image

        if isinstance(response, Image):
            return NormalizedResult(
                ok=True,
                artifacts=[
                    Artifact(
                        kind="image",
                        data=response.data if hasattr(response, "data") else None,
                        format=getattr(response, "format", "png"),
                    )
                ],
            )
    except ImportError:
        pass

    # JSON string (most tools via _json_response)
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return NormalizedResult(raw=response, parse_error=True)
        return _normalize_parsed(parsed, raw=response)

    # Already a dict or list (some internal paths)
    if isinstance(response, (dict, list)):
        return _normalize_parsed(response, raw=None)

    # Fallback
    return NormalizedResult(raw=str(response), parse_error=True)


def _normalize_parsed(parsed: Any, raw: str | None = None) -> NormalizedResult:
    if isinstance(parsed, dict):
        # For parse_error=true payloads, prefer the inner "raw" field
        # over the outer JSON wrapper so raw_contains inspects the
        # actual Wolfram output, not the JSON envelope.
        is_parse_error = parsed.get("parse_error", False)
        effective_raw = parsed.get("raw", raw) if is_parse_error else raw

        return NormalizedResult(
            ok=parsed.get("success", True),
            parsed=parsed,
            raw=effective_raw,
            output_text=(parsed.get("output_inputform") or parsed.get("output") or parsed.get("result")),
            warnings=_normalize_warnings(parsed),
            error_text=parsed.get("error"),
            parse_error=is_parse_error,
            artifacts=_extract_artifacts(parsed),
            meta={
                k: parsed[k]
                for k in (
                    "timing_ms",
                    "transport_status",
                    "execution_method",
                    "execution_path",
                )
                if k in parsed
            },
        )
    # Non-dict parsed (list, string, number)
    return NormalizedResult(ok=True, parsed=parsed, raw=raw, output_text=str(parsed))


def _normalize_warnings(parsed: dict[str, Any]) -> list[str]:
    """Normalize both warning formats used in this repo.

    1. String warnings: ["Power::infy: ..."]
    2. Dict message records: [{"type": "error", "tag": "Syntax::sntxf", "text": "..."}]
    """
    tags: list[str] = []
    for w in parsed.get("warnings", []):
        if isinstance(w, str):
            tags.append(w)
    for msg in parsed.get("messages", []):
        if isinstance(msg, dict):
            tag = msg.get("tag", "")
            text = msg.get("text", "")
            if tag:
                tags.append(f"{tag}: {text}" if text else tag)
    return tags


def _extract_artifacts(parsed: dict[str, Any]) -> list[Artifact]:
    """Extract artifact references from known response fields.

    Covers: image_path (execute_code graphics), rendered_image (graphics),
    file_path (export_data), path (save_notebook, export_notebook,
    export_graphics).
    """
    artifacts: list[Artifact] = []
    for key in ("image_path", "rendered_image", "file_path", "path"):
        val = parsed.get(key)
        if val and isinstance(val, str):
            artifacts.append(Artifact(kind="file", path=val))
    return artifacts
