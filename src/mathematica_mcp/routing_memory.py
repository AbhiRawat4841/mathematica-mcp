"""
Routing memory for execute_code.

Collects aggregate routing statistics (cohort counters, latency histograms,
error family frequencies, per-path transport outcomes) to improve
observability and enable conservative routing action.

Modes
-----
- off:     zero overhead, no I/O
- observe: records aggregate counters, persists to JSON periodically
- advise:  observe + generates routing hints + enables routing action (if gated)
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .constants import AttemptOutcome
from .wl_scan import scan_clean

logger = logging.getLogger("mathematica_mcp.routing_memory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENCY_BUCKET_BOUNDS = (50, 100, 250, 500, 1000, 5000)  # ms boundaries
_SCHEMA_VERSION = 2
_FLUSH_INTERVAL_S = 60.0
_FLUSH_BATCH_CAP = 200
_MAX_FLUSH_FAILURES = 3
_DECAY_HALF_LIFE_DAYS = 7.0

_CACHE_DIR = Path.home() / ".cache" / "mathematica-mcp"


def _get_storage_path() -> Path:
    base = Path(os.environ.get("MATHEMATICA_ROUTING_MEMORY_DIR", str(_CACHE_DIR)))
    return base / "routing_memory.json"


# ---------------------------------------------------------------------------
# Expression classification
# ---------------------------------------------------------------------------

_EXPR_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    (
        "frontend_dynamic",
        re.compile(r"\b(Manipulate|Dynamic|Animate|Slider|DynamicModule)\["),
    ),
    (
        "plot",
        re.compile(
            r"\b(Plot|Plot3D|ListPlot|Graphics|Show|ContourPlot|DensityPlot"
            r"|ParametricPlot|RegionPlot|ListLinePlot|BarChart|PieChart)\["
        ),
    ),
    (
        "symbolic_heavy",
        re.compile(
            r"\b(Integrate|Solve|DSolve|Sum|Series|Limit|Simplify|Expand|Factor|Apart|Together)\["
        ),
    ),
    (
        "numeric_heavy",
        re.compile(
            r"\b(NIntegrate|NSolve|NDSolve|FindRoot|NMinimize|NMaximize"
            r"|LinearSolve|Eigenvalues|SingularValueDecomposition)\["
        ),
    ),
    (
        "io",
        re.compile(r"\b(Export|Import|Get|Put|ReadList|BinaryReadList)\["),
    ),
]


def _classify_expression(code: str) -> str:
    """Classify Wolfram code into a routing-relevant category.

    Returns one of: ``"plot"``, ``"frontend_dynamic"``, ``"symbolic_heavy"``,
    ``"numeric_heavy"``, ``"io"``, ``"general"``.

    Strips strings and comments before matching so that content inside
    ``"Plot[...]"`` or ``(* Plot[...] *)`` does not trigger false matches.
    Falls back to ``"general"`` on malformed input.
    """
    scan = scan_clean(code)
    if not scan.ok:
        return "general"
    cleaned = scan.cleaned
    for category, pattern in _EXPR_CATEGORIES:
        if pattern.search(cleaned):
            return category
    return "general"


# ---------------------------------------------------------------------------
# Transport lease (lifecycle API)
# ---------------------------------------------------------------------------


@dataclass
class TransportLease:
    """Lease returned by ``begin_transport_attempt()``.

    Mutable: ``_completed`` is set internally by ``finish`` / ``abort``.
    """

    action: Literal["allow", "skip", "probe"]
    key: tuple[str, str] | None = None  # (route_variant, execution_path)
    _completed: bool = False


# ---------------------------------------------------------------------------
# Routing hint
# ---------------------------------------------------------------------------


@dataclass(order=True)
class RoutingHint:
    severity: int  # 1=infra_error, 2=timeout, 3=fallback (lower = more severe)
    specificity: int  # 1=expr-type, 2=aggregate (lower = more specific)
    label: str
    message: str


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
    # Per-path transport outcomes (aggregate cohorts only, fed by attempt telemetry)
    path_transport_outcomes: dict[str, dict[str, int]] = field(default_factory=dict)

    def total_calls(self) -> int:
        return self.ok_count + self.degraded_fallback_count + self.timeout_count + self.infra_error_count

    def apply_decay(self, factor: float) -> None:
        self.ok_count = int(self.ok_count * factor)
        self.degraded_fallback_count = int(self.degraded_fallback_count * factor)
        self.timeout_count = int(self.timeout_count * factor)
        self.infra_error_count = int(self.infra_error_count * factor)
        self.semantic_error_count = int(self.semantic_error_count * factor)
        self.unclassified_failure_count = int(self.unclassified_failure_count * factor)
        self.latency_hist = [int(b * factor) for b in self.latency_hist]
        self.path_counts = {k: int(v * factor) for k, v in self.path_counts.items() if int(v * factor) > 0}
        # Decay + prune path_transport_outcomes
        pruned_pto: dict[str, dict[str, int]] = {}
        for path, outcomes in self.path_transport_outcomes.items():
            pruned = {k: int(v * factor) for k, v in outcomes.items() if int(v * factor) > 0}
            if pruned:
                pruned_pto[path] = pruned
        self.path_transport_outcomes = pruned_pto

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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
        if self.path_transport_outcomes:
            d["path_transport_outcomes"] = self.path_transport_outcomes
        return d

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
            path_transport_outcomes=d.get("path_transport_outcomes", {}),
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

        # Breaker state (runtime-only, not persisted)
        # keyed by (route_variant, execution_path)
        from collections import deque

        self._recent_outcomes: dict[tuple[str, str], deque] = {}
        self._breaker_state: dict[tuple[str, str], str] = {}  # "closed"|"open"|"probe_in_flight"
        self._tripped_at: dict[tuple[str, str], float] = {}  # time.monotonic()
        self._breaker_cooldown_s: float = 60.0

        # Skip observability (runtime-only, not persisted)
        self._routing_action_skip_count: int = 0
        self._last_routing_skip_reason: str = ""
        self._last_routing_skip_at: float = 0.0

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
        expression_type: str | None = None,
    ) -> None:
        """Record final end-to-end result. O(1), thread-safe, never raises externally."""
        try:
            self._record_inner(
                profile,
                route_variant,
                execution_path,
                transport_status,
                latency_ms,
                error_families,
                expression_type,
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
        expression_type: str | None = None,
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

            # 7. Expr-type cohort (transport buckets only, no path_transport_outcomes)
            if expression_type:
                expr_key = f"{key}|{expression_type}"
                expr_cohort = self._cohorts.setdefault(expr_key, CohortStats())
                if transport_status == "timeout":
                    expr_cohort.timeout_count += 1
                elif transport_status == "degraded_fallback":
                    expr_cohort.degraded_fallback_count += 1
                elif transport_status == "infra_error":
                    expr_cohort.infra_error_count += 1
                else:
                    expr_cohort.ok_count += 1

            self._pending_count += 1

            # Check flush trigger
            should_flush = False
            elapsed = time.monotonic() - self._last_flush_ts
            if (elapsed >= _FLUSH_INTERVAL_S and self._pending_count > 0) or self._pending_count >= _FLUSH_BATCH_CAP:
                should_flush = True

        if should_flush:
            self.flush()

    # -- Attempt-level telemetry ---------------------------------------------

    def record_transport_attempt(
        self,
        profile: str,
        route_variant: str,
        execution_path: str,
        outcome: AttemptOutcome,
    ) -> None:
        """Record a single transport attempt (per-leg, not end-to-end).

        Feeds ``path_transport_outcomes`` on aggregate cohorts AND
        ``_recent_outcomes`` for the breaker.
        """
        try:
            with self._lock:
                key = self._cohort_key(profile, route_variant)
                cohort = self._cohorts.setdefault(key, CohortStats())
                path_outcomes = cohort.path_transport_outcomes.setdefault(execution_path, {})
                path_outcomes[outcome.value] = path_outcomes.get(outcome.value, 0) + 1

                # Feed breaker recent outcomes
                breaker_key = (route_variant, execution_path)
                if breaker_key not in self._recent_outcomes:
                    from collections import deque

                    self._recent_outcomes[breaker_key] = deque(maxlen=5)
                self._recent_outcomes[breaker_key].append(outcome)

                # Check if breaker should trip
                recent = self._recent_outcomes[breaker_key]
                if (
                    len(recent) >= 5
                    and all(o == AttemptOutcome.INFRA_ERROR for o in recent)
                    and self._breaker_state.get(breaker_key) != "probe_in_flight"
                ):
                    self._breaker_state[breaker_key] = "open"
                    self._tripped_at[breaker_key] = time.monotonic()
                elif outcome != AttemptOutcome.INFRA_ERROR:
                    # Any non-infra-error outcome clears the breaker
                    if breaker_key in self._breaker_state:
                        self._breaker_state[breaker_key] = "closed"
        except Exception:
            logger.debug("routing_memory record_transport_attempt failed", exc_info=True)

    # -- Lifecycle API -------------------------------------------------------

    def begin_transport_attempt(
        self,
        profile: str,  # noqa: ARG002
        route_variant: str,
        execution_path: str,
    ) -> TransportLease:
        """Return a transport lease.

        Checks breaker state. Returns action='skip' when breaker is tripped,
        action='probe' for half-open probe, action='allow' otherwise.

        Routing action requires ``routing_action == "compute_cli_skip"``
        AND ``route_variant == "compute"``. All other cases return 'allow'.
        """
        breaker_key = (route_variant, execution_path)

        # Import here to avoid circular import at module level
        from .config import FEATURES

        # Only compute addon_cli can be skipped
        if (
            FEATURES.routing_action != "compute_cli_skip"
            or route_variant != "compute"
        ):
            return TransportLease(action="allow", key=breaker_key)

        with self._lock:
            state = self._breaker_state.get(breaker_key, "closed")

            if state == "closed":
                return TransportLease(action="allow", key=breaker_key)

            if state == "open":
                elapsed = time.monotonic() - self._tripped_at.get(breaker_key, 0)
                if elapsed < self._breaker_cooldown_s:
                    # Within cooldown — skip
                    self._routing_action_skip_count += 1
                    self._last_routing_skip_reason = f"breaker_open:{execution_path}"
                    self._last_routing_skip_at = time.monotonic()
                    return TransportLease(action="skip", key=breaker_key)
                # Cooldown expired — half-open probe
                self._breaker_state[breaker_key] = "probe_in_flight"
                return TransportLease(action="probe", key=breaker_key)

            if state == "probe_in_flight":
                # Another caller is probing — skip
                self._routing_action_skip_count += 1
                self._last_routing_skip_reason = f"probe_in_flight:{execution_path}"
                self._last_routing_skip_at = time.monotonic()
                return TransportLease(action="skip", key=breaker_key)

        return TransportLease(action="allow", key=breaker_key)

    def finish_transport_attempt(
        self,
        lease: TransportLease,
        outcome: AttemptOutcome,
        profile: str = "",
    ) -> None:
        """Complete a transport attempt — single unified call.

        Handles BOTH:
        1. Persisted attempt telemetry (path_transport_outcomes + recent_outcomes)
        2. Breaker probe completion (for probe leases)

        Idempotent: second call on same lease is a no-op.
        """
        if lease._completed:
            return
        lease._completed = True

        if lease.key is not None:
            route_variant, execution_path = lease.key
            # Record attempt telemetry (persisted + breaker recent_outcomes)
            self.record_transport_attempt(
                profile or "unknown", route_variant, execution_path, outcome
            )

        # Handle probe completion
        if lease.action == "probe" and lease.key is not None:
            with self._lock:
                current = self._breaker_state.get(lease.key)
                if current == "probe_in_flight":
                    if outcome == AttemptOutcome.INFRA_ERROR:
                        # Retrip with fresh cooldown
                        self._breaker_state[lease.key] = "open"
                        self._tripped_at[lease.key] = time.monotonic()
                    else:
                        # Probe succeeded — close breaker
                        self._breaker_state[lease.key] = "closed"

    def abort_transport_attempt(self, lease: TransportLease) -> None:
        """Abort a probe that was never actually attempted.

        Reverts to open without recording a failure or resetting cooldown.
        No-op for non-probe leases. Idempotent.
        """
        if lease._completed:
            return
        lease._completed = True

        if lease.action == "probe" and lease.key is not None:
            with self._lock:
                current = self._breaker_state.get(lease.key)
                if current == "probe_in_flight":
                    # Revert to open — do NOT reset cooldown, do NOT record failure
                    self._breaker_state[lease.key] = "open"

    # -- Hints ---------------------------------------------------------------

    # Actionable transport legs for hint generation
    _HINTABLE_PATHS = frozenset({"addon_cli", "addon_notebook"})

    def get_routing_hints(self, profile: str) -> list[str]:
        """Return advisory routing hints. Returns [] in observe mode.

        Builds hints from two sources:
        1. Transport-path hints (from attempt telemetry)
        2. End-to-end hints (from final record cohort data)
        """
        if self.mode != "advise":
            return []

        hints: list[RoutingHint] = []

        with self._lock:
            for key, cohort in self._cohorts.items():
                parts = key.split("|")
                cohort_profile = parts[0]
                if cohort_profile != profile:
                    continue

                total = cohort.total_calls()
                if total < 5:
                    continue

                route_variant = parts[1] if len(parts) >= 2 else "unknown"
                expr_type = parts[2] if len(parts) >= 3 else None
                specificity = 1 if expr_type else 2
                label = f"{expr_type or route_variant} via {route_variant}" if expr_type else route_variant

                # End-to-end hints
                timeout_rate = cohort.timeout_count / total
                if timeout_rate > 0.3:
                    hints.append(RoutingHint(
                        severity=2, specificity=specificity, label=label,
                        message=f"{label}: {timeout_rate:.0%} timeout rate",
                    ))

                infra_rate = cohort.infra_error_count / total
                if infra_rate > 0.4:
                    hints.append(RoutingHint(
                        severity=1, specificity=specificity, label=label,
                        message=f"{label}: {infra_rate:.0%} infra error rate",
                    ))

                fallback_rate = cohort.degraded_fallback_count / total
                if fallback_rate > 0.4:
                    hints.append(RoutingHint(
                        severity=3, specificity=specificity, label=label,
                        message=f"{label}: {fallback_rate:.0%} fallback rate",
                    ))

                # Transport-path hints (aggregate cohorts only, actionable paths)
                if not expr_type:
                    for path, outcomes in cohort.path_transport_outcomes.items():
                        if path not in self._HINTABLE_PATHS:
                            continue
                        path_total = sum(outcomes.values())
                        if path_total < 10:
                            continue
                        # Include route_variant in label to avoid cross-variant conflation
                        path_label = f"{path} ({route_variant})"
                        infra_count = outcomes.get("infra_error", 0)
                        if infra_count / path_total > 0.3:
                            hints.append(RoutingHint(
                                severity=1, specificity=2, label=path_label,
                                message=f"{path_label}: {infra_count / path_total:.0%} infra error rate in transport attempts",
                            ))
                        timeout_count = outcomes.get("timeout", 0)
                        if timeout_count / path_total > 0.4:
                            hints.append(RoutingHint(
                                severity=2, specificity=2, label=path_label,
                                message=f"{path_label}: {timeout_count / path_total:.0%} timeout rate in transport attempts",
                            ))

        # Dedupe: if aggregate and expr-type both flag same issue, keep more specific
        seen: dict[str, RoutingHint] = {}
        for h in sorted(hints):
            dedupe_key = f"{h.label}:{h.severity}"
            if dedupe_key not in seen or h.specificity < seen[dedupe_key].specificity:
                seen[dedupe_key] = h
        deduped = sorted(seen.values())

        # Cap at 5 and render to strings
        return [h.message for h in deduped[:5]]

    # -- Recent error families -----------------------------------------------

    def get_recent_error_families(self, limit: int = 3, max_age_seconds: float = 86400) -> list[str]:
        """Return recently-seen error families sorted by last_seen desc.

        Filters out ``"other"`` unless no other families exist.
        Omits families older than *max_age_seconds*.
        """
        with self._lock:
            now = time.time()
            cutoff = now - max_age_seconds
            candidates = [
                (name, stats.last_seen)
                for name, stats in self._error_families.items()
                if stats.last_seen >= cutoff
            ]

        # Filter "other" unless it's the only candidate
        non_other = [(n, t) for n, t in candidates if n != "other"]
        if non_other:
            candidates = non_other

        candidates.sort(key=lambda x: -x[1])
        return [name for name, _ in candidates[:limit]]

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
            stats: dict[str, Any] = {
                "mode": self.mode,
                "persistence_enabled": self._persistence_enabled,
                "flush_failure_count": self._flush_failure_count,
                "total_calls": total,
                "cohort_count": len(self._cohorts),
                "top_cohorts": [{"key": k, "calls": n} for k, n in top_cohorts],
                "top_error_families": [{"family": k, "count": n} for k, n in top_families],
                "file_size_bytes": file_size,
            }
            # Routing action observability (runtime-only)
            if self._routing_action_skip_count > 0:
                stats["routing_action_skip_count"] = self._routing_action_skip_count
                stats["last_routing_skip_reason"] = self._last_routing_skip_reason
                stats["last_routing_skip_at"] = self._last_routing_skip_at
            return stats

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
            # Clear runtime-only breaker state
            self._recent_outcomes.clear()
            self._breaker_state.clear()
            self._tripped_at.clear()
            self._routing_action_skip_count = 0
            self._last_routing_skip_reason = ""
            self._last_routing_skip_at = 0.0
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
                k: {"count": v.count, "last_seen": v.last_seen} for k, v in self._error_families.items()
            },
        }

    def _write_atomic(self, data: dict[str, Any]) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._storage_path.parent),
            suffix=".tmp",
            prefix="routing_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, str(self._storage_path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Invalid routing memory file, starting fresh: %s", e)
            return

        # Schema check — accept v1 and v2 (v1 just lacks path_transport_outcomes)
        sv = raw.get("schema_version", 0)
        if sv not in (1, 2):
            logger.warning("Unknown routing memory schema version %s, starting fresh", sv)
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
