"""Shared transport outcome classification.

Single source of truth for both attempt-level and final-level transport
classification.  Prevents semantic drift between the two paths.

Refactored from server.py: _classify_transport, _extract_error_families.
"""

from __future__ import annotations

import re
from typing import Any

from .constants import AttemptOutcome

# ---------------------------------------------------------------------------
# Transport status labels (final, end-to-end)
# ---------------------------------------------------------------------------


class TransportStatus:
    OK = "ok"
    DEGRADED_FALLBACK = "degraded_fallback"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"


# ---------------------------------------------------------------------------
# Error family extraction (shared low-level helper)
# ---------------------------------------------------------------------------

ACTIONABLE_ERROR_FAMILIES = frozenset(
    {
        "Syntax",
        "Part",
        "Set",
        "Power",
        "Divide",
        "Recursion",
        "UnitConvert",
    }
)

_TAG_RE = re.compile(r"(\w+)::\w+")


def extract_error_families(response: dict[str, Any]) -> list[str]:
    """Best-effort semantic tag extraction from all available response fields."""
    families: set[str] = set()

    # Notebook path: structured messages with tag/text/type dicts
    for msg in response.get("messages", []):
        if isinstance(msg, dict) and msg.get("type") == "error":
            tag = msg.get("tag", "")
            family = tag.split("::")[0] if "::" in tag else ""
            if family in ACTIONABLE_ERROR_FAMILIES:
                families.add(family)
            elif family and family != "General":
                families.add("other")

    # Kernel/CLI path: raw warning strings
    for w in response.get("warnings", []):
        if isinstance(w, str):
            for match in _TAG_RE.findall(w):
                if match in ACTIONABLE_ERROR_FAMILIES:
                    families.add(match)
                elif match != "General":
                    families.add("other")

    # Fallback: scan error field
    if not families:
        error_text = response.get("error", "")
        if isinstance(error_text, str):
            for match in _TAG_RE.findall(error_text):
                if match in ACTIONABLE_ERROR_FAMILIES:
                    families.add(match)

    # Last resort: scan output ONLY if success=False AND error is empty
    if not families and response.get("success") is False and not response.get("error"):
        output_text = response.get("output", "")
        if isinstance(output_text, str):
            for match in _TAG_RE.findall(output_text):
                if match in ACTIONABLE_ERROR_FAMILIES:
                    families.add(match)

    return sorted(families)


# ---------------------------------------------------------------------------
# Final transport classification (end-to-end)
# ---------------------------------------------------------------------------


def classify_final_transport(response: dict[str, Any], *, fell_back: bool) -> str:
    """Classify transport status from final result + extracted error families."""
    if response.get("timed_out"):
        return TransportStatus.TIMEOUT
    if response.get("success") is False:
        # Semantic families found → transport worked, math failed
        if response.get("error_families"):
            return TransportStatus.DEGRADED_FALLBACK if fell_back else TransportStatus.OK
        return TransportStatus.INFRA_ERROR
    if fell_back:
        return TransportStatus.DEGRADED_FALLBACK
    return TransportStatus.OK


# ---------------------------------------------------------------------------
# Attempt-level classification
# ---------------------------------------------------------------------------


def classify_attempt_outcome(
    result: dict[str, Any] | None,
    exc: Exception | None = None,
) -> AttemptOutcome:
    """Map a raw transport attempt result to a typed AttemptOutcome.

    This is the single source of truth for attempt classification.
    Uses the same low-level helpers (error family extraction, timeout detection)
    as final classification to prevent semantic drift.

    Parameters
    ----------
    result : dict or None
        The raw result dict from the transport attempt.  None if an exception
        was raised before a result was obtained.
    exc : Exception or None
        The exception raised during the attempt, if any.
    """
    # Exception before result → infra error
    if exc is not None:
        return AttemptOutcome.INFRA_ERROR

    if result is None:
        return AttemptOutcome.INFRA_ERROR

    # Non-dict results (rare string returns) → treat as infra error
    if not isinstance(result, dict):
        return AttemptOutcome.INFRA_ERROR

    # Timeout
    if result.get("timed_out"):
        return AttemptOutcome.TIMEOUT

    # Failure: explicit success=False OR presence of "error" key (mirrors server fallback predicate)
    if result.get("success") is False or "error" in result:
        error_families = result.get("error_families") or extract_error_families(result)
        if error_families:
            return AttemptOutcome.SEMANTIC_ERROR
        return AttemptOutcome.INFRA_ERROR

    return AttemptOutcome.OK
