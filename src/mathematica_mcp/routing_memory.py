"""
Observe-only routing memory for execute_code.

Collects aggregate routing statistics (cohort counters, latency histograms,
error family frequencies) to improve observability.  No raw traces, no SQLite,
no background threads.

Modes
-----
- off:     zero overhead, no I/O
- observe: records aggregate counters, persists to JSON periodically
- advise:  observe + generates routing hints (Phase 2, gated)
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("mathematica_mcp.routing_memory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENCY_BUCKET_BOUNDS = (50, 100, 250, 500, 1000, 5000)  # ms boundaries
_SCHEMA_VERSION = 1
_FLUSH_INTERVAL_S = 60.0
_FLUSH_BATCH_CAP = 200
_MAX_FLUSH_FAILURES = 3
_DECAY_HALF_LIFE_DAYS = 7.0

_CACHE_DIR = Path.home() / ".cache" / "mathematica-mcp"

def _get_storage_path() -> Path:
    base = Path(os.environ.get("MATHEMATICA_ROUTING_MEMORY_DIR", str(_CACHE_DIR)))
    return base / "routing_memory.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def _latency_bucket(ms: int) -> int:
    """Return histogram bucket index for a latency value."""
    for i, bound in enumerate(LATENCY_BUCKET_BOUNDS):
        if ms < bound:
            return i
    return len(LATENCY_BUCKET_BOUNDS)  # >5000 bucket


def _empty_hist() -> list[int]:
    return [0] * (len(LATENCY_BUCKET_BOUNDS) + 1)


@dataclass
class CohortStats:
    ok_count: int = 0
    degraded_fallback_count: int = 0
    timeout_count: int = 0
    infra_error_count: int = 0
    semantic_error_count: int = 0
    unclassified_failure_count: int = 0
    latency_hist: list[int] = field(default_factory=_empty_hist)
    path_counts: dict[str, int] = field(default_factory=dict)

    def total_calls(self) -> int:
        return (
            self.ok_count
            + self.degraded_fallback_count
            + self.timeout_count
            + self.infra_error_count
        )

    def apply_decay(self, factor: float) -> None:
        self.ok_count = int(self.ok_count * factor)
        self.degraded_fallback_count = int(self.degraded_fallback_count * factor)
        self.timeout_count = int(self.timeout_count * factor)
        self.infra_error_count = int(self.infra_error_count * factor)
        self.semantic_error_count = int(self.semantic_error_count * factor)
        self.unclassified_failure_count = int(self.unclassified_failure_count * factor)
        self.latency_hist = [int(b * factor) for b in self.latency_hist]
        self.path_counts = {k: int(v * factor) for k, v in self.path_counts.items() if int(v * factor) > 0}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok_count,
            "degraded_fallback": self.degraded_fallback_count,
            "timeout": self.timeout_count,
            "infra_error": self.infra_error_count,
            "semantic_error": self.semantic_error_count,
            "unclassified_failure": self.unclassified_failure_count,
            "latency_hist": self.latency_hist,
            "path_counts": self.path_counts,
            "total_calls": self.total_calls(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CohortStats:
        return cls(
            ok_count=d.get("ok", 0),
            degraded_fallback_count=d.get("degraded_fallback", 0),
            timeout_count=d.get("timeout", 0),
            infra_error_count=d.get("infra_error", 0),
            semantic_error_count=d.get("semantic_error", 0),
            unclassified_failure_count=d.get("unclassified_failure", 0),
            latency_hist=d.get("latency_hist", _empty_hist()),
            path_counts=d.get("path_counts", {}),
        )


@dataclass
class ErrorFamilyStats:
    count: int = 0
    last_seen: float = 0.0

    def apply_decay(self, factor: float) -> None:
        self.count = int(self.count * factor)


# ---------------------------------------------------------------------------
# RoutingMemory
# ---------------------------------------------------------------------------

class RoutingMemory:
    def __init__(self, mode: str, *, storage_path: Path | None = None):
        self.mode = mode
        self._storage_path = storage_path or _get_storage_path()
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()

        # Cohort data: key = "profile|route_variant"
        self._cohorts: dict[str, CohortStats] = {}
        self._error_families: dict[str, ErrorFamilyStats] = {}
        self._last_decay_ts: float = time.time()

        # Flush bookkeeping
        self._pending_count = 0
        self._last_flush_ts = time.monotonic()
        self._flush_generation = 0
        self._last_written_gen = -1
        self._flush_failure_count = 0
        self._persistence_enabled = True

        # Load persisted data
        self._load()

    @staticmethod
    def _cohort_key(profile: str, route_variant: str) -> str:
        return f"{profile}|{route_variant}"

    # -- Recording ----------------------------------------------------------

    def record(
        self,
        profile: str,
        route_variant: str,
        execution_path: str,
        transport_status: str,
        latency_ms: int,
        error_families: list[str],
    ) -> None:
        """Increment aggregate counters. O(1), thread-safe, never raises externally."""
        try:
            self._record_inner(
                profile, route_variant, execution_path,
                transport_status, latency_ms, error_families,
            )
        except Exception:
            logger.debug("routing_memory record failed", exc_info=True)

    def _record_inner(
        self,
        profile: str,
        route_variant: str,
        execution_path: str,
        transport_status: str,
        latency_ms: int,
        error_families: list[str],
    ) -> None:
        with self._lock:
            # Decay old state BEFORE recording new event so fresh events
            # are not immediately aged after idle periods.
            self._maybe_decay()

            key = self._cohort_key(profile, route_variant)
            cohort = self._cohorts.setdefault(key, CohortStats())

            # 1. Exactly one transport bucket
            if transport_status == "timeout":
                cohort.timeout_count += 1
            elif transport_status == "degraded_fallback":
                cohort.degraded_fallback_count += 1
            elif transport_status == "infra_error":
                cohort.infra_error_count += 1
            else:
                cohort.ok_count += 1

            # 2. Semantic error count (independent)
            if error_families:
                cohort.semantic_error_count += 1

            # 3. Unclassified failure (infra_error with no families)
            if transport_status == "infra_error" and not error_families:
                cohort.unclassified_failure_count += 1

            # 4. Path counts
            cohort.path_counts[execution_path] = cohort.path_counts.get(execution_path, 0) + 1

            # 5. Latency histogram
            bucket = _latency_bucket(latency_ms)
            cohort.latency_hist[bucket] += 1

            # 6. Error family stats (once per unique family per call)
            now = time.time()
            for family in set(error_families):
                fstats = self._error_families.setdefault(family, ErrorFamilyStats())
                fstats.count += 1
                fstats.last_seen = now

            self._pending_count += 1

            # Check flush trigger
            should_flush = False
            elapsed = time.monotonic() - self._last_flush_ts
            if (elapsed >= _FLUSH_INTERVAL_S and self._pending_count > 0) or self._pending_count >= _FLUSH_BATCH_CAP:
                should_flush = True

        if should_flush:
            self.flush()

    # -- Hints (Phase 2 stub) -----------------------------------------------

    def get_routing_hints(self, profile: str) -> list[str]:  # noqa: ARG002
        """Return advisory routing hints. Returns [] in observe mode."""
        return []

    # -- Stats ---------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total = sum(c.total_calls() for c in self._cohorts.values())
            top_cohorts = sorted(
                ((k, c.total_calls()) for k, c in self._cohorts.items()),
                key=lambda x: -x[1],
            )[:5]
            top_families = sorted(
                ((k, v.count) for k, v in self._error_families.items()),
                key=lambda x: -x[1],
            )[:5]
            file_size = 0
            try:
                if self._storage_path.exists():
                    file_size = self._storage_path.stat().st_size
            except OSError:
                pass
            return {
                "mode": self.mode,
                "persistence_enabled": self._persistence_enabled,
                "flush_failure_count": self._flush_failure_count,
                "total_calls": total,
                "cohort_count": len(self._cohorts),
                "top_cohorts": [{"key": k, "calls": n} for k, n in top_cohorts],
                "top_error_families": [{"family": k, "count": n} for k, n in top_families],
                "file_size_bytes": file_size,
            }

    # -- Persistence ---------------------------------------------------------

    def flush(self) -> None:
        """Persist a snapshot. May drop older overlapping snapshots (generation check)."""
        if not self._persistence_enabled:
            return
        try:
            with self._lock:
                # Apply decay during flush so long-running processes don't keep stale counts
                self._maybe_decay()
                snapshot = self._serialize()
                gen = self._flush_generation
                self._flush_generation += 1
                self._pending_count = 0
                self._last_flush_ts = time.monotonic()

            with self._flush_lock:
                if gen >= self._last_written_gen:
                    self._write_atomic(snapshot)
                    self._last_written_gen = gen
                    self._flush_failure_count = 0
        except Exception:
            self._flush_failure_count += 1
            if self._flush_failure_count >= _MAX_FLUSH_FAILURES:
                self._persistence_enabled = False
                logger.warning(
                    "Routing memory persistence disabled after %d consecutive failures",
                    self._flush_failure_count,
                )
            else:
                logger.debug("routing_memory flush failed", exc_info=True)

    def clear(self) -> None:
        """Reset in-memory state and remove persisted state, atomic w.r.t. flush."""
        with self._lock:
            self._cohorts.clear()
            self._error_families.clear()
            self._last_decay_ts = time.time()
            self._pending_count = 0
            # Advance generation past any in-flight snapshot to suppress stale writes
            self._flush_generation += 1
            # Re-enable persistence (clear should be a full reset)
            self._persistence_enabled = True
            self._flush_failure_count = 0
        with self._flush_lock:
            # Advance last_written_gen so any older snapshot is suppressed
            self._last_written_gen = self._flush_generation
            try:
                if self._storage_path.exists():
                    self._storage_path.unlink()
            except OSError:
                pass

    def shutdown(self) -> None:
        self.flush()

    def _serialize(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "last_decay_ts": self._last_decay_ts,
            "cohorts": {k: v.to_dict() for k, v in self._cohorts.items()},
            "error_families": {
                k: {"count": v.count, "last_seen": v.last_seen}
                for k, v in self._error_families.items()
            },
        }

    def _write_atomic(self, data: dict[str, Any]) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._storage_path.parent), suffix=".tmp", prefix="routing_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, str(self._storage_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Invalid routing memory file, starting fresh: %s", e)
            return

        # Schema check
        if raw.get("schema_version", 0) != _SCHEMA_VERSION:
            logger.warning("Unknown routing memory schema version %s, starting fresh", raw.get("schema_version"))
            return

        self._last_decay_ts = raw.get("last_decay_ts", time.time())

        for key, cdata in raw.get("cohorts", {}).items():
            self._cohorts[key] = CohortStats.from_dict(cdata)

        for family, fdata in raw.get("error_families", {}).items():
            self._error_families[family] = ErrorFamilyStats(
                count=fdata.get("count", 0),
                last_seen=fdata.get("last_seen", 0.0),
            )

        # Apply exponential decay
        self._maybe_decay()

    def _maybe_decay(self) -> None:
        now = time.time()
        elapsed_days = (now - self._last_decay_ts) / 86400.0
        if elapsed_days < 1.0:
            return
        factor = 0.5 ** (elapsed_days / _DECAY_HALF_LIFE_DAYS)
        for cohort in self._cohorts.values():
            cohort.apply_decay(factor)
        for fstats in self._error_families.values():
            fstats.apply_decay(factor)
        self._last_decay_ts = now


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_CURRENT_INSTANCE: RoutingMemory | None = None
_atexit_registered = False


def _atexit_shutdown() -> None:
    if _CURRENT_INSTANCE is not None:
        _CURRENT_INSTANCE.shutdown()


def init(mode: str, *, storage_path: Path | None = None) -> RoutingMemory | None:
    """Initialize routing memory. Returns None on failure or if mode='off'.

    Note: if an instance with the same mode already exists, it is returned
    unchanged (storage_path is ignored). Call _reset_for_tests() first
    if you need to change the storage path.
    """
    global _CURRENT_INSTANCE, _atexit_registered

    if mode == "off":
        return None

    # Re-entry: return existing if same mode, shutdown old if different
    if _CURRENT_INSTANCE is not None:
        if _CURRENT_INSTANCE.mode == mode:
            return _CURRENT_INSTANCE
        _CURRENT_INSTANCE.shutdown()

    try:
        instance = RoutingMemory(mode, storage_path=storage_path)
    except Exception as e:
        logger.warning("Failed to initialize routing memory: %s", e)
        return None

    _CURRENT_INSTANCE = instance

    if not _atexit_registered:
        atexit.register(_atexit_shutdown)
        _atexit_registered = True

    return instance


def _reset_for_tests() -> None:
    """Test-only: shutdown current instance and clear singleton state."""
    global _CURRENT_INSTANCE
    if _CURRENT_INSTANCE is not None:
        _CURRENT_INSTANCE.shutdown()
    _CURRENT_INSTANCE = None
