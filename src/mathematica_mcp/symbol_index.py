"""Version-scoped symbol index for fast name lookups without subprocess calls.

On first use, builds an in-memory list of all ``System`` symbol names via a
single ``wolframscript`` call.  Subsequent lookups are pure Python string
matching — no subprocess needed.  The index is keyed by kernel version so it
auto-rebuilds if the Wolfram Language installation changes.

A separate metadata cache stores per-symbol info (usage, options, attributes)
fetched lazily on demand.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mathematica_mcp.symbol_index")

_index_lock = threading.Lock()
_system_symbols: List[str] = []
_system_symbols_lower: List[tuple[str, str]] = []  # (lower_name, original_name)
_kernel_version: str = ""

_metadata_lock = threading.Lock()
_symbol_metadata: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


def _build_index_sync() -> bool:
    """Build the symbol index by querying wolframscript once."""
    global _system_symbols, _system_symbols_lower, _kernel_version

    from .lazy_wolfram_tools import _find_wolframscript

    wolframscript = _find_wolframscript()
    if not wolframscript:
        return False

    code = 'StringRiffle[Join[{ToString[$VersionNumber]}, Names["System`*"]], "\\n"]'

    try:
        result = subprocess.run(
            [wolframscript, "-code", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("wolframscript exited with code %d", result.returncode)
            return False

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return False

        version = lines[0].strip()
        symbols = [s.strip() for s in lines[1:] if s.strip()]

        with _index_lock:
            _kernel_version = version
            _system_symbols = symbols
            _system_symbols_lower = [(s.lower(), s) for s in symbols]

        logger.info(
            "Symbol index built: %d symbols for version %s", len(symbols), version
        )
        return True
    except Exception as e:
        logger.warning("Failed to build symbol index: %s", e)
        return False


def ensure_index() -> bool:
    """Ensure the symbol index is populated.  Build on first call."""
    with _index_lock:
        if _system_symbols:
            return True
    return _build_index_sync()


def invalidate() -> None:
    """Clear the index and metadata cache (e.g. after kernel version change)."""
    global _system_symbols, _system_symbols_lower, _kernel_version
    with _index_lock:
        _system_symbols = []
        _system_symbols_lower = []
        _kernel_version = ""
    with _metadata_lock:
        _symbol_metadata.clear()


# ---------------------------------------------------------------------------
# Name search (pure Python)
# ---------------------------------------------------------------------------


def search(query: str, *, max_results: int = 20) -> List[str]:
    """Return symbol names matching *query* (case-insensitive).

    Results are ordered: exact match first, then prefix matches, then
    substring matches.  No subprocess is spawned.
    """
    if not ensure_index():
        return []

    query_lower = query.lower()

    with _index_lock:
        exact: list[str] = []
        starts: list[str] = []
        contains: list[str] = []

        for lower, original in _system_symbols_lower:
            if lower == query_lower:
                exact.append(original)
            elif lower.startswith(query_lower):
                starts.append(original)
            elif query_lower in lower:
                contains.append(original)

        return (exact + starts + contains)[:max_results]


def get_version() -> str:
    """Return the cached kernel version string."""
    with _index_lock:
        return _kernel_version


# ---------------------------------------------------------------------------
# Per-symbol metadata cache
# ---------------------------------------------------------------------------


def get_cached_metadata(symbol: str) -> Optional[Dict[str, Any]]:
    """Return cached metadata for *symbol*, or ``None`` on miss."""
    with _metadata_lock:
        return _symbol_metadata.get(symbol)


def cache_metadata(symbol: str, metadata: Dict[str, Any]) -> None:
    """Merge *metadata* into the cache entry for *symbol*.

    Existing keys are preserved; new keys are added.  This prevents
    a usage-only hydration from clobbering a full metadata entry
    (or vice versa).
    """
    with _metadata_lock:
        existing = _symbol_metadata.get(symbol)
        if existing:
            existing.update(metadata)
        else:
            _symbol_metadata[symbol] = dict(metadata)
