"""In-memory computation journal for execute_code.

Bounded ring buffer that records recent computation metadata.
Records raw normalized responses BEFORE response-detail filtering
so the journal captures canonical results, not view-layer truncations.
"""

from __future__ import annotations

import time
from typing import Any


class ComputationJournal:
    """Bounded in-memory ring buffer of recent computation entries."""

    def __init__(self, max_entries: int = 10):
        self._entries: list[dict[str, Any]] = []
        self._max_entries = max_entries

    def record(
        self,
        code: str,
        output: str,
        *,
        success: bool,
        timing_ms: int,
        route_variant: str = "",
        execution_path: str = "",
        transport_status: str = "",
        error_families: list[str] | None = None,
        timed_out: bool = False,
        from_cache: bool = False,
        session_id: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "code_preview": code[:100],
            "output_preview": output[:100],
            "success": success,
            "timing_ms": timing_ms,
            "route_variant": route_variant,
            "execution_path": execution_path,
            "transport_status": transport_status,
            "error_families": error_families or [],
            "timed_out": timed_out,
            "from_cache": from_cache,
        }
        if session_id is not None:
            entry["session_id"] = session_id
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def get_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": len(self._entries),
            "entries": self._entries,
        }

    def clear(self) -> None:
        self._entries.clear()
