"""Version-scoped symbol index for fast name lookups without subprocess calls.

On first use, builds an in-memory list of all ``System`` symbol names via a
single ``wolframscript`` call.  Subsequent lookups are pure Python string
matching — no subprocess needed.  The index is keyed by kernel version so it
auto-rebuilds if the Wolfram Language installation changes.

A disk cache (``~/.cache/mathematica-mcp/symbols/``) persists the index
across process restarts.  The cache key is derived from the resolved
wolframscript binary identity (path + mtime + size), so it auto-invalidates
when the installation changes.

A separate metadata cache stores per-symbol info (usage, options, attributes)
fetched lazily on demand.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("mathematica_mcp.symbol_index")

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_index_lock = threading.Lock()
_system_symbols: list[str] = []
_system_symbols_lower: list[tuple[str, str]] = []  # (lower_name, original_name)
_kernel_version: str = ""

_metadata_lock = threading.Lock()
_symbol_metadata: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Singleflight state machine
# ---------------------------------------------------------------------------


class _BuildState(enum.IntEnum):
    IDLE = 0
    BUILDING = 1
    BUILT = 2


_build_cond = threading.Condition()
_build_state = _BuildState.IDLE
_build_generation: int = 0  # incremented on invalidate()

# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

_SYMBOLS_CACHE_DIR = Path(
    os.environ.get(
        "MATHEMATICA_MCP_SYMBOLS_CACHE_DIR",
        str(Path.home() / ".cache" / "mathematica-mcp" / "symbols"),
    )
)


def _get_binary_identity() -> tuple[str, int, int] | None:
    """Return (real_path, mtime_ns, size) for the wolframscript binary.

    Resolves symlinks so the identity reflects the actual binary, not a
    stable wrapper whose metadata may not change across installations.
    """
    from .lazy_wolfram_tools import _find_wolframscript

    ws = _find_wolframscript()
    if not ws:
        return None
    try:
        real_path = os.path.realpath(ws)
        st = os.stat(real_path)
        return (real_path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _cache_key(real_path: str, mtime_ns: int, size: int) -> str:
    """Deterministic cache key from binary identity."""
    raw = f"{real_path}:{mtime_ns}:{size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load_from_disk_cache() -> bool:
    """Try to load the symbol index from disk cache. Returns True on hit."""
    global _system_symbols, _system_symbols_lower, _kernel_version

    identity = _get_binary_identity()
    if identity is None:
        return False

    real_path, mtime_ns, size = identity
    key = _cache_key(real_path, mtime_ns, size)
    cache_dir = _SYMBOLS_CACHE_DIR
    cache_file = cache_dir / f"{key}.json"

    if not cache_file.exists():
        return False

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        # Validate staleness guards
        if data.get("_binary_mtime_ns") != mtime_ns:
            cache_file.unlink(missing_ok=True)
            return False
        if data.get("_binary_size") != size:
            cache_file.unlink(missing_ok=True)
            return False

        symbols = data.get("symbols", [])
        version = data.get("kernel_version", "")
        if not symbols:
            return False

        with _index_lock:
            _kernel_version = version
            _system_symbols = symbols
            _system_symbols_lower = [(s.lower(), s) for s in symbols]

        logger.info(
            "Symbol index loaded from disk cache: %d symbols for version %s",
            len(symbols),
            version,
        )
        return True
    except (json.JSONDecodeError, OSError, KeyError) as e:
        logger.warning("Corrupt symbol cache entry %s: %s", cache_file, e)
        cache_file.unlink(missing_ok=True)
        return False


def _save_to_disk_cache() -> None:
    """Persist the current in-memory index to disk cache (atomic write)."""
    identity = _get_binary_identity()
    if identity is None:
        return

    with _index_lock:
        if not _system_symbols:
            return
        symbols = list(_system_symbols)
        version = _kernel_version

    real_path, mtime_ns, size = identity
    key = _cache_key(real_path, mtime_ns, size)

    cache_dir = _SYMBOLS_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"

    cache_data = {
        "_wolframscript_path": real_path,
        "_binary_mtime_ns": mtime_ns,
        "_binary_size": size,
        "kernel_version": version,
        "symbols": symbols,
    }

    # Atomic write: temp file + rename
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(cache_dir), suffix=".tmp", prefix="sym_"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        os.replace(tmp_path, str(cache_file))
        logger.debug("Symbol index saved to disk cache: %s", cache_file)
    except OSError as e:
        logger.warning("Failed to write symbol cache: %s", e)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def clear_disk_cache() -> int:
    """Remove all cached symbol index files. Returns count of files removed."""
    cache_dir = _SYMBOLS_CACHE_DIR
    if not cache_dir.exists():
        return 0
    count = 0
    for f in cache_dir.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


def _build_index_sync() -> bool:
    """Build the symbol index by querying wolframscript once.

    This is the pure build function — no cache logic. It spawns a
    wolframscript subprocess to retrieve all System` symbol names.
    """
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

        logger.info("Symbol index built: %d symbols for version %s", len(symbols), version)
        return True
    except Exception as e:
        logger.warning("Failed to build symbol index: %s", e)
        return False


def ensure_index() -> bool:
    """Ensure the symbol index is populated.

    Orchestrates: in-memory check → disk cache → wolframscript build → save.
    Uses a singleflight pattern: only one thread builds at a time, others wait.
    A generation counter prevents stale builds from publishing after invalidation.
    """
    global _build_state, _build_generation, _system_symbols, _system_symbols_lower, _kernel_version

    with _build_cond:
        # Fast path: symbols already populated (works for both normal builds
        # and test fixtures that populate _system_symbols directly)
        with _index_lock:
            if _system_symbols:
                _build_state = _BuildState.BUILT
                return True

        # Another thread is building: wait for it
        if _build_state == _BuildState.BUILDING:
            _build_cond.wait(timeout=35)  # slightly > wolframscript timeout
            with _index_lock:
                return bool(_system_symbols)

        # We're the builder — record generation and transition
        gen = _build_generation
        _build_state = _BuildState.BUILDING

    # Outside the lock: try disk cache, then wolframscript build
    try:
        success = _load_from_disk_cache()
        if not success:
            success = _build_index_sync()
            if success:
                _save_to_disk_cache()

        with _build_cond:
            if _build_generation == gen:
                # No invalidation during build — publish results
                _build_state = _BuildState.BUILT if success else _BuildState.IDLE
            else:
                # Invalidation happened during build — clear stale data
                # that _load_from_disk_cache/_build_index_sync already wrote
                with _index_lock:
                    _system_symbols = []
                    _system_symbols_lower = []
                    _kernel_version = ""
                _build_state = _BuildState.IDLE
                success = False
            _build_cond.notify_all()
        return success
    except Exception:
        with _build_cond:
            _build_state = _BuildState.IDLE
            _build_cond.notify_all()
        return False


def invalidate() -> None:
    """Clear the in-memory index, metadata cache, and reset build state.

    Does NOT touch the disk cache — use ``clear_disk_cache()`` for that.
    Increments the generation counter to prevent stale in-flight builds
    from publishing results.
    """
    global _system_symbols, _system_symbols_lower, _kernel_version
    global _build_state, _build_generation

    with _build_cond:
        _build_state = _BuildState.IDLE
        _build_generation += 1
        _build_cond.notify_all()

    with _index_lock:
        _system_symbols = []
        _system_symbols_lower = []
        _kernel_version = ""
    with _metadata_lock:
        _symbol_metadata.clear()


# ---------------------------------------------------------------------------
# Name search (pure Python)
# ---------------------------------------------------------------------------


def search(query: str, *, max_results: int = 20) -> list[str]:
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


def get_cached_metadata(symbol: str) -> dict[str, Any] | None:
    """Return cached metadata for *symbol*, or ``None`` on miss."""
    with _metadata_lock:
        return _symbol_metadata.get(symbol)


def cache_metadata(symbol: str, metadata: dict[str, Any]) -> None:
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
