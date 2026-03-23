"""
Disk-based cache for notebook extraction results.

Caches converted notebook content to avoid re-conversion across
process restarts.  Keyed on: content hash + backend + options.
Invalidated when the source file changes (mtime + size check).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("mathematica_mcp.disk_cache")

# Default cache directory
_CACHE_DIR = Path.home() / ".cache" / "mathematica-mcp" / "notebooks"


def get_cache_dir() -> Path:
    """Return (and ensure exists) the disk cache directory."""
    cache_dir = Path(os.environ.get("MATHEMATICA_MCP_CACHE_DIR", str(_CACHE_DIR)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_key(
    path: str,
    mtime_ns: int,
    file_size: int,
    backend: str,
    options_hash: str,
) -> str:
    """Generate a stable cache key from file identity + extraction params."""
    raw = f"{path}|{mtime_ns}|{file_size}|{backend}|{options_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _options_hash(**kwargs: Any) -> str:
    """Hash extraction options to include in cache key."""
    # Sort for determinism
    canonical = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def get_cached(
    path: str,
    backend: str,
    **options: Any,
) -> dict[str, Any] | None:
    """Retrieve cached notebook extraction result, or None if miss/stale."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        return None

    try:
        stat = resolved.stat()
    except OSError:
        return None

    key = _cache_key(
        str(resolved),
        stat.st_mtime_ns,
        stat.st_size,
        backend,
        _options_hash(**options),
    )
    cache_file = get_cache_dir() / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        # Verify staleness guard
        if data.get("_mtime_ns") != stat.st_mtime_ns:
            cache_file.unlink(missing_ok=True)
            return None
        if data.get("_file_size") != stat.st_size:
            cache_file.unlink(missing_ok=True)
            return None
        logger.debug("Disk cache hit for %s (backend=%s)", path, backend)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Corrupt cache entry %s: %s", cache_file, e)
        cache_file.unlink(missing_ok=True)
        return None


def put_cached(
    path: str,
    backend: str,
    data: dict[str, Any],
    **options: Any,
) -> None:
    """Store notebook extraction result to disk cache."""
    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return

    key = _cache_key(
        str(resolved),
        stat.st_mtime_ns,
        stat.st_size,
        backend,
        _options_hash(**options),
    )

    # Add staleness guards to data
    cache_data = dict(data)
    cache_data["_mtime_ns"] = stat.st_mtime_ns
    cache_data["_file_size"] = stat.st_size
    cache_data["_source_path"] = str(resolved)
    cache_data["_backend"] = backend

    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{key}.json"

    # Atomic write: write to temp, then rename
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp", prefix="mcp_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        os.replace(tmp_path, str(cache_file))
        logger.debug("Disk cache write for %s (backend=%s)", path, backend)
    except OSError as e:
        logger.warning("Failed to write cache: %s", e)
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def clear_cache() -> int:
    """Remove all cached entries. Returns count of files removed."""
    cache_dir = get_cache_dir()
    count = 0
    for f in cache_dir.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count
