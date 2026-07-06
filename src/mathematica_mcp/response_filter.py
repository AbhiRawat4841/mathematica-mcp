"""Pure response filtering for execute_code payload shaping.

No runtime singleton imports.  All external data (cache epoch, routing hints)
passed as parameters to keep this module decoupled from server state.
"""

from __future__ import annotations

from typing import Any

from .wl_scan import count_top_level_braces

# ---------------------------------------------------------------------------
# Large output summarisation
# ---------------------------------------------------------------------------


def _summarize_large_output(output: str, threshold: int = 4000) -> dict[str, Any] | None:
    """Produce a structured summary for outputs exceeding *threshold* chars.

    Returns ``None`` if the output is below threshold.
    """
    if len(output) <= threshold:
        return None

    summary: dict[str, Any] = {
        "original_length": len(output),
        "truncated_preview": output[:500],
        "tail_preview": output[-200:],
        "summary": f"Output is {len(output)} characters. First 500 and last 200 shown.",
    }

    # Attempt list element count via balanced-brace scan
    element_count = count_top_level_braces(output)
    if element_count is not None:
        summary["element_count"] = element_count
        summary["summary"] += f" List with {element_count} top-level elements."

    return summary


# ---------------------------------------------------------------------------
# Fields kept/stripped per detail level
# ---------------------------------------------------------------------------

# compact keeps these keys if present in the response
_COMPACT_KEEP = frozenset(
    {
        "success",
        "status",
        "evaluation_complete",  # frontend pending/complete contract
        "waited_seconds",
        "next_step",  # pending/error recovery guidance
        "message",
        "output",
        "timing_ms",
        "executed_output_target",
        "executed_mode",
        "notebook_id",
        "cell_id",
        "transport_status",
        "error_families",
        "timed_out",
        "from_cache",
        "error",
        "auto_routed",  # interactive auto-route note; its audience is compact-mode lean clients
        "error_analysis",  # keep guidance + retry_with even in compact
        "available_followups",
        "state_delta",
        "rendered_image",
        "is_graphics",
        "output_inputform",  # kept temporarily for graphics swap, may be stripped after
    }
)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


_FAILURE_STATUSES = frozenset({"timeout", "error", "executed_with_errors", "notebook_error"})


def _is_failure(response: dict[str, Any]) -> bool:
    """True when the response signals failure. Compact mode keeps the full shape
    for these so agents can recover (transport_status, error_families, messages,
    error_analysis, output_preview). Notebook timeout/error shapes carry no
    ``success`` key, so failure is inferred from any of these signals."""
    return (
        response.get("success") is False
        or bool(response.get("error"))
        or bool(response.get("timed_out"))
        or bool(response.get("has_errors"))
        or response.get("status") in _FAILURE_STATUSES
    )


def _filter_response(
    response: dict[str, Any],
    detail: str,
    *,
    cache_epoch: int | None = None,
    routing_hints: list[str] | None = None,
    summarize_large: bool = True,
) -> dict[str, Any]:
    """Shape the response payload based on *detail* level.

    - ``"standard"``: exact identity — zero modifications.
    - ``"compact"``: keep essential fields, swap graphics placeholder,
      summarise large output, strip empty fields. Failures are exempt: they
      pass through as the full identity shape so recovery info survives.
    - ``"verbose"``: identity + ``detail_level``.
    - ``"diagnostic"``: identity + ``detail_level`` + optional extras.
    - Unknown values: treated as ``"standard"``.
    """
    if detail == "standard" or detail not in ("compact", "verbose", "diagnostic"):
        return response

    if detail == "verbose":
        return {**response, "detail_level": "verbose"}

    if detail == "diagnostic":
        result = {**response, "detail_level": "diagnostic"}
        if cache_epoch is not None:
            result["cache_epoch"] = cache_epoch
        if routing_hints is not None:
            result["routing_hints"] = routing_hints
        return result

    # --- compact ---
    # Failures keep the full identity shape so transport_status, error_families,
    # messages, warnings and error_analysis all survive for recovery.
    if _is_failure(response):
        return response

    result: dict[str, Any] = {}
    for key in _COMPACT_KEEP:
        if key in response:
            result[key] = response[key]

    # Graphics fix: prefer output_inputform over rendered-image placeholder
    if result.get("is_graphics") and "output_inputform" in result:
        output = result.get("output", "")
        if isinstance(output, str) and output.startswith("[Graphics rendered to image:"):
            result["output"] = result["output_inputform"]
    # Strip output_inputform from compact output (kept only for the swap above)
    result.pop("output_inputform", None)

    # Large output summarisation. Skipped when a downstream cursor paginator will
    # serve the full output (lean path); summarising first drops ~16K chars the
    # paginator could have returned page by page.
    if summarize_large:
        output = result.get("output", "")
        if isinstance(output, str):
            summary = _summarize_large_output(output)
            if summary is not None:
                result["output"] = summary["truncated_preview"]
                result["output_summary"] = summary

    # Strip empty fields ("", [], {}, None) - always keep success, output and
    # state_delta (an empty output/state_delta is itself a result, and callers
    # index response["state_delta"] directly).
    result = {
        k: v for k, v in result.items() if k in ("success", "output", "state_delta") or v not in ("", [], {}, None)
    }

    return result
